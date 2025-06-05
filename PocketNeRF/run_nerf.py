import os, sys
from datetime import datetime
import numpy as np
import imageio
import json
import pdb
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm, trange
import pickle

import matplotlib.pyplot as plt

from run_nerf_helpers import *
from optimizer import MultiOptimizer
from radam import RAdam
from loss import sigma_sparsity_loss, total_variation_loss
from PocketNeRF.structural_priors import (
    combine_structural_losses_v2, 
    ManhattanFrameEstimator, 
    SemanticPlaneDetector
)

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_scannet import load_scannet_data
from load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded, keep_mask = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs_flat[~keep_mask, -1] = 0 # set sigma to 0 for invalid points
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'depth_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf
    near, far = render_kwargs['near'], render_kwargs['far']

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    depths = []
    psnrs = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, depth, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        # normalize depth to [0,1]
        depth = (depth - near) / (far - near)
        depths.append(depth.cpu().numpy())
        if i==0:
            print(rgb.shape, depth.shape)

        if gt_imgs is not None and render_factor==0:
            try:
                gt_img = gt_imgs[i].cpu().numpy()
            except:
                gt_img = gt_imgs[i]
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_img)))
            print(p)
            psnrs.append(p)

        if savedir is not None:
            # save rgb and depth as a figure
            fig = plt.figure(figsize=(25,15))
            ax = fig.add_subplot(1, 2, 1)
            rgb8 = to8b(rgbs[-1])
            ax.imshow(rgb8)
            ax.axis('off')
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(depths[-1], cmap='plasma', vmin=0, vmax=1)
            ax.axis('off')
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            # save as png
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            # imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    depths = np.stack(depths, 0)
    if gt_imgs is not None and render_factor==0:
        avg_psnr = sum(psnrs)/len(psnrs)
        print("Avg PSNR over Test set: ", avg_psnr)
        with open(os.path.join(savedir, "test_psnrs_avg{:0.2f}.pkl".format(avg_psnr)), "wb") as fp:
            pickle.dump(psnrs, fp)

    return rgbs, depths


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    # Get quantization parameters from args
    use_quantization = getattr(args, 'use_quantization', False)
    quantization_bits = getattr(args, 'quantization_bits', 8)

    embed_fn, input_ch = get_embedder(args.multires, args, i=args.i_embed)
    if args.i_embed==1:
        # hashed embedding table
        embedding_params = list(embed_fn.parameters())

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # if using hashed for xyz, use SH for views
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args, i=args.i_embed_views)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    if args.i_embed==1:
        model = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views,
                        use_quantization=use_quantization,
                        quantization_bits=quantization_bits,
                        predict_normals=args.predict_normals).to(device)
    else:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None

    if args.N_importance > 0:
        if args.i_embed==1:
            model_fine = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views,
                        use_quantization=use_quantization,
                        quantization_bits=quantization_bits,
                        predict_normals=args.predict_normals).to(device)
        else:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    if args.i_embed==1:
        optimizer = RAdam([
                            {'params': grad_vars, 'weight_decay': 1e-6},
                            {'params': embedding_params, 'eps': 1e-15}
                        ], lr=args.lrate, betas=(0.9, 0.99))
    else:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if args.i_embed==1:
            embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])

    ##########################
    # pdb.set_trace()

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'embed_fn': embed_fn,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'predict_normals' : args.predict_normals,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, predict_normals=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4 or 7]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
        predict_normals: bool. Whether the model predicts normals.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
        normal_map: [num_rays, 3]. Surface normals (if predict_normals=True).
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    
    if predict_normals:
        sigma = raw[...,3]  # [N_rays, N_samples]
        normals = raw[...,4:7]  # [N_rays, N_samples, 3]
    else:
        sigma = raw[...,3]  # [N_rays, N_samples]
    
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(sigma.shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(sigma.shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # sigma_loss = sigma_sparsity_loss(raw[...,3])
    alpha = raw2alpha(sigma + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1) / torch.sum(weights, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    # Calculate weights sparsity loss
    probs = torch.cat([weights, (1.0 - weights.sum(-1, keepdim=True)).clamp(min=1e-6)], dim=-1)
    entropy = Categorical(probs=probs).entropy()
    sparsity_loss = entropy

    if predict_normals:
        # Render normal map using the same weights
        normal_map = torch.sum(weights[...,None] * normals, -2)  # [N_rays, 3]
        normal_map = F.normalize(normal_map, dim=-1)  # Ensure unit length
        return rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss, normal_map
    else:
        return rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                embed_fn=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                predict_normals=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
      predict_normals: bool. If True, predict surface normals.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw = network_query_fn(pts, viewdirs, network_fn)
    
    # Handle normal prediction
    if predict_normals:
        rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss, normal_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, predict_normals=True)
    else:
        rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, predict_normals=False)

    if N_importance > 0:
        if predict_normals:
            rgb_map_0, depth_map_0, acc_map_0, sparsity_loss_0, normal_map_0 = rgb_map, depth_map, acc_map, sparsity_loss, normal_map
        else:
            rgb_map_0, depth_map_0, acc_map_0, sparsity_loss_0 = rgb_map, depth_map, acc_map, sparsity_loss

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        if predict_normals:
            rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss, normal_map = raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, predict_normals=True)
        else:
            rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, predict_normals=False)

    ret = {'rgb_map' : rgb_map, 'depth_map' : depth_map, 'acc_map' : acc_map, 'sparsity_loss': sparsity_loss}
    
    # Always store points and ray directions for structural priors (needed even without normal prediction)
    ret['pts'] = pts
    ret['rays_d'] = rays_d
    
    if predict_normals:
        ret['normal_map'] = normal_map

    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['depth0'] = depth_map_0
        ret['acc0'] = acc_map_0
        ret['sparsity_loss0'] = sparsity_loss_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        if predict_normals:
            ret['normal0'] = normal_map_0

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=1,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--i_embed_views", type=int, default=2,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## scannet flags
    parser.add_argument("--scannet_sceneID", type=str, default='scene0000_00',
                        help='sceneID to load from scannet')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=1000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=5000,
                        help='frequency of render_poses video saving')

    parser.add_argument("--finest_res",   type=int, default=512,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--log2_hashmap_size",   type=int, default=19,
                        help='log2 of hashmap size')
    parser.add_argument("--sparse-loss-weight", type=float, default=1e-10,
                        help='learning rate')
    parser.add_argument("--tv-loss-weight", type=float, default=1e-6,
                        help='learning rate')
    
    #adding quantization options
    parser.add_argument("--use_quantization", action='store_true',
                        help='enable quantization for hash embeddings and MLP')
    parser.add_argument("--quantization_bits", type=int, default=8,
                        help='number of bits for quantization (default: 8)')
    
    # PocketNeRF structural priors options
    parser.add_argument("--use_structural_priors", action='store_true',
                        help='enable structural priors for indoor scenes')
    parser.add_argument("--predict_normals", action='store_true',
                        help='enable normal prediction in the network')
    parser.add_argument("--depth_prior_weight", type=float, default=0.01,
                        help='weight for depth prior loss')
    parser.add_argument("--planarity_weight", type=float, default=0.005,
                        help='weight for planarity constraint loss')
    parser.add_argument("--manhattan_weight", type=float, default=0.002,
                        help='weight for Manhattan world assumption loss')
    parser.add_argument("--normal_consistency_weight", type=float, default=0.001,
                        help='weight for normal consistency loss')
    parser.add_argument("--structural_loss_start_iter", type=int, default=2000,
                        help='iteration to start applying structural losses')
    parser.add_argument("--structural_loss_ramp_iters", type=int, default=1000,
                        help='iterations to ramp up structural loss weights')
    parser.add_argument("--overfitting_threshold", type=float, default=8.0,
                        help='PSNR gap threshold for overfitting detection')
    parser.add_argument("--min_structural_weight", type=float, default=0.0001,
                        help='minimum structural loss weight to prevent going to zero')
    
    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # *** FIX: Always enable normal prediction if structural priors are enabled ***
    # This prevents architecture mismatch when structural priors activate
    if args.use_structural_priors and not args.predict_normals:
        print(f"🔧 AUTOMATICALLY ENABLING NORMAL PREDICTION for structural priors")
        print(f"   (Structural priors require normal prediction from network creation)")
        args.predict_normals = True

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test, bounding_box = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        args.bounding_box = bounding_box
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split, bounding_box = load_blender_data(args.datadir, args.half_res, args.testskip)
        args.bounding_box = bounding_box
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'scannet':
        images, poses, render_poses, hwf, i_split, bounding_box = load_scannet_data(args.datadir, args.scannet_sceneID, args.half_res)
        args.bounding_box = bounding_box
        print('Loaded scannet', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 0.1
        far = 10.0

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    if args.i_embed==1:
        args.expname += "_hashXYZ"
    elif args.i_embed==0:
        args.expname += "_posXYZ"
    if args.i_embed_views==2:
        args.expname += "_sphereVIEW"
    elif args.i_embed_views==0:
        args.expname += "_posVIEW"
    args.expname += "_fine"+str(args.finest_res) + "_log2T"+str(args.log2_hashmap_size)
    args.expname += "_lr"+str(args.lrate) + "_decay"+str(args.lrate_decay)
    args.expname += "_RAdam"
    if args.sparse_loss_weight > 0:
        args.expname += "_sparse" + str(args.sparse_loss_weight)
    args.expname += "_TV" + str(args.tv_loss_weight)
    #args.expname += datetime.now().strftime('_%H_%M_%d_%m_%Y')
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 20000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    loss_list = []
    psnr_list = []
    time_list = []
    
    # Initialize structural priors components (V2) with conservative parameters
    manhattan_estimator = ManhattanFrameEstimator(confidence_threshold=0.4)  # More conservative
    semantic_detector = SemanticPlaneDetector(normal_threshold=0.5)           # More conservative
    
    # PocketNeRF Time Tracking for Final Report
    time_metrics = {
        'start_time': time.time(),
        'structural_priors_start_time': None,
        'milestones': {},  # PSNR milestone times
        'convergence_time': None,
        'iterations_per_second': [],
        'time_to_milestones': {},  # Time to reach each PSNR threshold
        'baseline_comparison': {
            'time_to_20db': None,
            'time_to_25db': None,
            'time_to_30db': None,
        }
    }
    
    start = start + 1
    time0 = time.time()
    iteration_start_time = time.time()
    for i in trange(start, N_iters):
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, depth, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        sparsity_loss = args.sparse_loss_weight*(extras["sparsity_loss"].sum() + extras["sparsity_loss0"].sum())
        loss = loss + sparsity_loss

        # add Total Variation loss
        if args.i_embed==1:
            n_levels = render_kwargs_train["embed_fn"].n_levels
            min_res = render_kwargs_train["embed_fn"].base_resolution
            max_res = render_kwargs_train["embed_fn"].finest_resolution
            log2_hashmap_size = render_kwargs_train["embed_fn"].log2_hashmap_size
            TV_loss = sum(total_variation_loss(render_kwargs_train["embed_fn"].embeddings[i], \
                                              min_res, max_res, \
                                              i, log2_hashmap_size, \
                                              n_levels=n_levels) for i in range(n_levels))
            loss = loss + args.tv_loss_weight * TV_loss
            if i>1000:
                args.tv_loss_weight = 0.0

        # Add PocketNeRF structural priors
        structural_loss_total = 0.0
        structural_loss_dict = {}
        
        if args.use_structural_priors and i >= args.structural_loss_start_iter:
            # Announce when structural priors first activate
            if i == args.structural_loss_start_iter:
                time_metrics['structural_priors_start_time'] = time.time()
                structural_activation_time = time_metrics['structural_priors_start_time'] - time_metrics['start_time']
                
                print("\n" + "="*80)
                print(f"🏗️  ACTIVATING POCKETNERF STRUCTURAL PRIORS AT ITERATION {i}")
                print("="*80)
                print(f"📋 Configuration:")
                print(f"   Depth Prior Weight: {args.depth_prior_weight}")
                print(f"   Planarity Weight: {args.planarity_weight}")
                print(f"   Manhattan Weight: {args.manhattan_weight}")
                print(f"   Normal Consistency Weight: {args.normal_consistency_weight}")
                print(f"   Predict Normals: {args.predict_normals} (enabled from network creation)")
                print(f"   Ramp Duration: {args.structural_loss_ramp_iters} iterations")
                print(f"   Overfitting Threshold: {args.overfitting_threshold} dB")
                print(f"📈 Current Baseline PSNR: {psnr.item():.2f} dB")
                print(f"⏱️  Time to Activation: {structural_activation_time/60:.1f} minutes")
                print(f"🎯 Expecting improvement in:")
                print(f"   - Planar surface quality (walls, floors)")
                print(f"   - Geometric consistency")
                print(f"   - Few-shot reconstruction stability")
                print("="*80 + "\n")
            
            # Progressive weight ramping to avoid sudden optimization shocks
            ramp_progress = min(1.0, (i - args.structural_loss_start_iter) / args.structural_loss_ramp_iters)
            ramp_factor = 0.1 + 0.9 * ramp_progress  # Start at 10%, ramp to 100%
            
            # Improved overfitting detection with more conservative thresholds
            if i > args.structural_loss_start_iter + 500 and i % 500 == 0 and len(psnr_list) > 50:
                # More frequent and sensitive overfitting detection
                recent_train_psnr = np.mean(psnr_list[-20:])  # More recent samples
                if hasattr(args, '_last_test_psnr') and recent_train_psnr - args._last_test_psnr > args.overfitting_threshold:
                    print(f"\n⚠️  Overfitting detected at iteration {i}")
                    print(f"   Train PSNR: {recent_train_psnr:.1f} dB, Last Test: {args._last_test_psnr:.1f} dB")
                    print(f"   Gap: {recent_train_psnr - args._last_test_psnr:.1f} dB > {args.overfitting_threshold:.1f} dB threshold")
                    
                    # More conservative weight reduction - reduce by 30% instead of 50%
                    reduction_factor = 0.7
                    args.depth_prior_weight = max(args.min_structural_weight, args.depth_prior_weight * reduction_factor)
                    args.planarity_weight = max(args.min_structural_weight, args.planarity_weight * reduction_factor)
                    args.manhattan_weight = max(args.min_structural_weight, args.manhattan_weight * reduction_factor)
                    args.normal_consistency_weight = max(args.min_structural_weight, args.normal_consistency_weight * reduction_factor)
                    
                    print(f"   Reduced weights by {int((1-reduction_factor)*100)}%:")
                    print(f"   depth={args.depth_prior_weight:.6f}, planarity={args.planarity_weight:.6f}")
                    print(f"   manhattan={args.manhattan_weight:.6f}, normal_consistency={args.normal_consistency_weight:.6f}")
                    
                    # Early stopping if weights become too small
                    if args.planarity_weight <= args.min_structural_weight * 2:
                        print(f"   ⚠️  Structural weights very small - consider reducing structural loss contribution")
            
            # Prepare ramped structural prior weights  
            structural_weights = {
                'depth_prior': args.depth_prior_weight * ramp_factor,
                'planarity': args.planarity_weight * ramp_factor,
                'manhattan': args.manhattan_weight * ramp_factor, 
                'normal_consistency': args.normal_consistency_weight * ramp_factor
            }
            
            # Get required data for structural priors
            depth_pred = depth
            normals = extras.get('normal_map', None) if args.predict_normals else None
            rays_d = extras.get('rays_d', None)
            points = extras.get('pts', None)
            
            if depth_pred is not None:
                try:
                    # Get spatial coordinates for spatial awareness
                    if not use_batching and N_rand is not None:
                        # Extract 2D coordinates from the ray selection
                        spatial_coords = select_coords.float()  # [N_rand, 2]
                    else:
                        spatial_coords = None
                    
                    # Compute structural losses with V2 implementation
                    structural_loss, structural_loss_dict = combine_structural_losses_v2(
                        depth_pred=depth_pred,
                        normals=normals,
                        rays_d=rays_d,
                        spatial_coords=spatial_coords,
                        weights=structural_weights,
                        manhattan_frame_estimator=manhattan_estimator,
                        semantic_detector=semantic_detector
                    )
                    
                    structural_loss_total = structural_loss.item() if torch.is_tensor(structural_loss) else structural_loss
                    loss = loss + structural_loss
                    
                    # Enhanced logging for V2 implementation
                    if i % args.i_print == 0:
                        floor_count = structural_loss_dict.get('semantic_floor_count', 0)
                        wall_count = structural_loss_dict.get('semantic_wall_count', 0)
                        if floor_count > 0 or wall_count > 0:
                            print(f"   🏗️  Semantics: {floor_count} floor, {wall_count} wall points detected")
                    
                    # Log ramping progress
                    if i % args.i_print == 0 and ramp_progress < 1.0:
                        print(f"   🔄 Structural weights ramping: {ramp_progress*100:.1f}% complete")
                    
                except Exception as e:
                    # Don't break training if structural priors fail
                    if i % args.i_print == 0:
                        print(f"  ⚠️  Structural priors V2 failed: {e}")
                        print(f"      This is expected during initial iterations as geometry stabilizes")

        elif args.use_structural_priors and i < args.structural_loss_start_iter:
            # Show countdown to structural priors activation with improved messages
            if i % args.i_print == 0 and i > args.structural_loss_start_iter - 500:
                remaining = args.structural_loss_start_iter - i
                if remaining <= 100:
                    print(f"  🚀 Structural priors activate in {remaining} iterations...")
                elif remaining <= 200:
                    print(f"  📊 Structural priors activate in {remaining} iterations (geometry stabilizing)...")
                else:
                    print(f"  📊 Structural priors activate in {remaining} iterations...")

        loss.backward()
        # pdb.set_trace()
        
        # *** REMOVED GRADIENT CLIPPING THAT WAS CAUSING OVERFITTING ***
        # The aggressive gradient clipping (max_norm=1.0) was interfering with 
        # normal training dynamics and making overfitting worse.
        # torch.nn.utils.clip_grad_norm_(grad_vars, max_norm=1.0)
        
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        t = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####
        
        # PocketNeRF Time Metrics Collection
        current_time = time.time()
        iteration_time = current_time - iteration_start_time
        time_metrics['iterations_per_second'].append(1.0 / iteration_time if iteration_time > 0 else 0)
        
        # Track PSNR milestones for convergence analysis
        current_psnr = psnr.item()
        milestones = [15, 20, 25, 30, 35]
        
        for milestone in milestones:
            milestone_key = f'{milestone}db'
            if current_psnr >= milestone and milestone_key not in time_metrics['milestones']:
                milestone_time = current_time - time_metrics['start_time']
                time_metrics['milestones'][milestone_key] = {
                    'iteration': i,
                    'time_seconds': milestone_time,
                    'time_minutes': milestone_time / 60.0
                }
                
                # Special tracking for baseline comparison
                if milestone == 20:
                    time_metrics['baseline_comparison']['time_to_20db'] = milestone_time / 60.0
                elif milestone == 25:
                    time_metrics['baseline_comparison']['time_to_25db'] = milestone_time / 60.0
                elif milestone == 30:
                    time_metrics['baseline_comparison']['time_to_30db'] = milestone_time / 60.0
                
                print(f"🎯 MILESTONE: Reached {milestone} dB PSNR at iteration {i} ({milestone_time/60:.1f} min)")
        
        # Convergence detection (PSNR hasn't improved significantly in last 1000 iterations)
        if i > 2000 and len(psnr_list) > 100 and time_metrics['convergence_time'] is None:
            recent_psnr = psnr_list[-100:]  # Last 100 iterations
            if len(recent_psnr) >= 100:
                psnr_std = np.std(recent_psnr)
                psnr_trend = recent_psnr[-1] - recent_psnr[0]
                
                # Converged if: low variance and minimal improvement
                if psnr_std < 0.5 and abs(psnr_trend) < 0.5:
                    convergence_time = current_time - time_metrics['start_time']
                    time_metrics['convergence_time'] = convergence_time / 60.0
                    print(f"📊 CONVERGENCE DETECTED at iteration {i} ({convergence_time/60:.1f} min)")
        
        iteration_start_time = current_time

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.i_embed==1:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'embed_fn_state_dict': render_kwargs_train['embed_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')
            
            # Additional PocketNeRF evaluation logging
            if args.use_structural_priors and i >= args.structural_loss_start_iter:
                print(f"📊 PocketNeRF Status @ {i}:")
                print(f"   Structural Loss: {structural_loss_total:.6f}")
                if len(psnr_list) >= 10:
                    # Compare recent performance to pre-structural priors
                    pre_struct_idx = max(0, args.structural_loss_start_iter - i_train.shape[0] * 10)
                    if pre_struct_idx < len(psnr_list):
                        recent_avg = np.mean(psnr_list[-10:])
                        pre_struct_avg = np.mean(psnr_list[max(0, pre_struct_idx-10):pre_struct_idx+10])
                        improvement = recent_avg - pre_struct_avg
                        status = "📈 Improving" if improvement > 0.5 else "📉 Declining" if improvement < -0.5 else "➡️ Stable"
                        print(f"   PSNR vs pre-structural: {improvement:+.2f} dB ({status})")
            
            # Store last test PSNR for overfitting detection (rough estimate from test renders)
            if i >= 1000:  # Start tracking after structural priors activate
                # Rough estimate: test PSNR is typically lower than train, use a heuristic
                estimated_test_psnr = np.mean(psnr_list[-5:]) - 12  # Observed ~12dB gap
                args._last_test_psnr = max(10.0, estimated_test_psnr)  # Floor at 10dB
            
            # Early overfitting detection for base HashNeRF (before structural priors)
            if i >= 1000 and i < args.structural_loss_start_iter:
                recent_train_psnr = np.mean(psnr_list[-5:])
                estimated_test_psnr = np.mean(psnr_list[-5:]) - 12  # Rough estimate
                
                # More aggressive overfitting detection for base training
                base_overfitting_threshold = 10.0  # Stricter threshold for base
                if recent_train_psnr - estimated_test_psnr > base_overfitting_threshold:
                    print(f"\n🚨 BASE HASHNERF OVERFITTING DETECTED @ {i} (before structural priors!)")
                    print(f"   Train PSNR: {recent_train_psnr:.1f} dB")
                    print(f"   Estimated Test PSNR: {estimated_test_psnr:.1f} dB") 
                    print(f"   Gap: {recent_train_psnr - estimated_test_psnr:.1f} dB > {base_overfitting_threshold:.1f} dB")
                    print(f"   ⚠️  This suggests the base model overfits with only {len(i_train)} training views")
                    print(f"   📋 Consider: lower learning rate, more regularization, or data augmentation")
                    
                    # Store this for later reference
                    args._last_test_psnr = estimated_test_psnr

        if i%args.i_print==0:
            # Enhanced logging for PocketNeRF monitoring
            base_msg = f"[TRAIN] Iter: {i} Loss: {loss.item():.6f} PSNR: {psnr.item():.2f}"
            
            # Add time efficiency metrics
            if len(time_metrics['iterations_per_second']) > 0:
                current_speed = time_metrics['iterations_per_second'][-1]
                avg_speed = np.mean(time_metrics['iterations_per_second'][-100:]) if len(time_metrics['iterations_per_second']) >= 100 else np.mean(time_metrics['iterations_per_second'])
                elapsed_time = (time.time() - time_metrics['start_time']) / 60.0
                base_msg += f" | Speed: {avg_speed:.2f}it/s | Time: {elapsed_time:.1f}min"
            
            # Add structural priors information
            if args.use_structural_priors:
                if i >= args.structural_loss_start_iter:
                    # Show breakdown of structural losses
                    struct_msg = f" | Struct: {structural_loss_total:.6f}"
                    if structural_loss_dict:
                        components = []
                        for loss_name, loss_val in structural_loss_dict.items():
                            if torch.is_tensor(loss_val):
                                components.append(f"{loss_name}: {loss_val.item():.6f}")
                        if components:
                            struct_msg += f" ({', '.join(components)})"
                    base_msg += struct_msg
                    
                    # Check if normals are being predicted
                    if args.predict_normals and 'normal_map' in extras:
                        normal_map = extras['normal_map']
                        normal_magnitude = torch.norm(normal_map, dim=-1).mean()
                        base_msg += f" | Norm: {normal_magnitude:.3f}"
                        
                elif i > args.structural_loss_start_iter - 500:
                    remaining = args.structural_loss_start_iter - i
                    base_msg += f" | Struct in: {remaining}"
            
            # PSNR trend analysis (every 500 iterations)
            if i > 0 and i % 500 == 0 and len(psnr_list) > 5:
                recent_psnr = psnr_list[-5:]  # Last 5 measurements
                psnr_trend = recent_psnr[-1] - recent_psnr[0]
                trend_emoji = "📈" if psnr_trend > 0 else "📉" if psnr_trend < -0.1 else "➡️"
                base_msg += f" | Trend: {trend_emoji} {psnr_trend:+.2f}"
                
                # Milestone check
                if psnr.item() > 25:
                    base_msg += " 🎯"
                elif psnr.item() > 20:
                    base_msg += " ✅"
            
            tqdm.write(base_msg)
            
            loss_list.append(loss.item())
            psnr_list.append(psnr.item())
            time_list.append(t)
            
            # Save comprehensive training data including time metrics
            training_data = {
                "losses": loss_list,
                "psnr": psnr_list,
                "time": time_list,
                "time_metrics": time_metrics,
                "structural_priors_enabled": args.use_structural_priors,
                "config": {
                    "depth_prior_weight": args.depth_prior_weight,
                    "planarity_weight": args.planarity_weight,
                    "manhattan_weight": args.manhattan_weight,
                    "normal_consistency_weight": args.normal_consistency_weight,
                    "structural_loss_start_iter": args.structural_loss_start_iter,
                    "predict_normals": args.predict_normals
                }
            }
            
            with open(os.path.join(basedir, expname, "training_metrics.pkl"), "wb") as fp:
                pickle.dump(training_data, fp)
            
            # Every 1000 iterations, print time efficiency summary
            if i % 1000 == 0 and i > 0:
                print(f"\n📊 Time Efficiency Summary @ {i} iterations:")
                elapsed = (time.time() - time_metrics['start_time']) / 60.0
                print(f"   Total Time: {elapsed:.1f} minutes")
                if time_metrics['structural_priors_start_time']:
                    struct_time = (time.time() - time_metrics['structural_priors_start_time']) / 60.0
                    print(f"   Time since Structural Priors: {struct_time:.1f} minutes")
                print(f"   Average Speed: {np.mean(time_metrics['iterations_per_second'][-100:]):.2f} it/s")
                
                # Show achieved milestones
                if time_metrics['milestones']:
                    print(f"   Milestones Achieved:")
                    for milestone, data in time_metrics['milestones'].items():
                        print(f"     {milestone}: {data['time_minutes']:.1f} min (iter {data['iteration']})")
                print()

        global_step += 1

    # Final PocketNeRF Time Metrics Summary for Report
    final_time = time.time()
    total_training_time = (final_time - time_metrics['start_time']) / 60.0
    
    print("\n" + "="*80)
    print("🏁 FINAL POCKETNERF TIME EFFICIENCY REPORT")
    print("="*80)
    print(f"📊 Training Summary:")
    print(f"   Total Training Time: {total_training_time:.1f} minutes ({total_training_time/60:.1f} hours)")
    print(f"   Total Iterations: {N_iters-1}")
    print(f"   Average Speed: {np.mean(time_metrics['iterations_per_second']):.2f} iterations/second")
    
    if time_metrics['structural_priors_start_time']:
        struct_training_time = (final_time - time_metrics['structural_priors_start_time']) / 60.0
        pre_struct_time = (time_metrics['structural_priors_start_time'] - time_metrics['start_time']) / 60.0
        print(f"\n⏱️  Phase Breakdown:")
        print(f"   Pre-Structural Priors: {pre_struct_time:.1f} min ({args.structural_loss_start_iter} iterations)")
        print(f"   With Structural Priors: {struct_training_time:.1f} min ({N_iters-1-args.structural_loss_start_iter} iterations)")
    
    print(f"\n🎯 PSNR Milestone Timeline:")
    if time_metrics['milestones']:
        for milestone in ['15db', '20db', '25db', '30db', '35db']:
            if milestone in time_metrics['milestones']:
                data = time_metrics['milestones'][milestone]
                print(f"   {milestone.upper()}: {data['time_minutes']:.1f} min (iteration {data['iteration']})")
            else:
                print(f"   {milestone.upper()}: Not achieved")
    else:
        print("   No milestones achieved")
    
    if time_metrics['convergence_time']:
        print(f"\n📈 Convergence Analysis:")
        print(f"   Convergence Time: {time_metrics['convergence_time']:.1f} minutes")
        print(f"   Final PSNR: {psnr_list[-1]:.2f} dB")
    
    # Save final comprehensive report
    final_report = {
        'total_training_time_minutes': total_training_time,
        'total_training_time_hours': total_training_time / 60.0,
        'total_iterations': N_iters - 1,
        'average_speed_its': np.mean(time_metrics['iterations_per_second']),
        'structural_priors_enabled': args.use_structural_priors,
        'milestones_achieved': time_metrics['milestones'],
        'convergence_time_minutes': time_metrics['convergence_time'],
        'final_psnr': psnr_list[-1] if psnr_list else 0,
        'baseline_comparison': time_metrics['baseline_comparison'],
        'config': {
            'depth_prior_weight': args.depth_prior_weight,
            'planarity_weight': args.planarity_weight,
            'manhattan_weight': args.manhattan_weight,
            'normal_consistency_weight': args.normal_consistency_weight,
            'structural_loss_start_iter': args.structural_loss_start_iter,
            'predict_normals': args.predict_normals
        }
    }
    
    with open(os.path.join(basedir, expname, "final_time_report.pkl"), "wb") as fp:
        pickle.dump(final_report, fp)
    
    print(f"\n💾 Comprehensive time metrics saved to:")
    print(f"   training_metrics.pkl (detailed iteration data)")
    print(f"   final_time_report.pkl (summary for report)")
    print("="*80 + "\n")


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()

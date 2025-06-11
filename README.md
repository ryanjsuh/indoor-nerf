# PocketNeRF

Fast-converging neural radiance fields for indoor reconstruction from few-shot mobile images. Built on Instant-NGP's hash-encoded NeRFs with structural priors for faster guided training and content-aware quantization for model compression.

## Getting Started

Prerequisites:

```bash
# clone repo
git clone <repository-url>
cd indoor-nerf

# create + activate venv
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r PocketNeRF/requirements.txt
```

Basic usage:

```bash
# Train on a scene using script
./run.sh <data_subpath>

# Or train directly w/ specific config
cd PocketNeRF
python run_nerf.py --config configs/chair.txt --finest_res 1024

# For iPhone prepocessing: see notebooks/iphone_raw_preprocessing.ipynb
```

To deactivate:

```bash
deactivate
```

## Background

PocketNeRF presents a lightweight pipeline for rapid, mobile-ready reconstruction of indoor environments from sparse smartphone images. Built on HashNeRF (a PyTorch implementation of Instant-NGP), it introduces two orthogonal improvements: structural priors based on Manhattan-world assumptions and semantic plane detection that impose geometric constraints during optimization, and Adversarial Content-Aware Quantization (A-CAQ) which learns scene-specific bitwidths to compress hash tables and MLP weights without degrading visual quality. These enhancements reduce training time by an order of magnitude and enable interactive, photorealistic rendering on consumer mobile devices.

## Authors

[Lucas Brennan](mailto:leba@stanford.edu)

[Aaron Jin](https://github.com/aaronkjin)

[Ryan Suh](https://github.com/ryanjsuh)

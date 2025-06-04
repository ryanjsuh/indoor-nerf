#!/usr/bin/env bash
set -e

# PocketNeRF BASELINE (no structural priors) for comparison
# Run with: ./run_baseline.sh fewshot_iphone/norcliffe_common_room

if [ -z "$1" ]; then
  echo "Usage: $0 <data_subpath>"
  exit 1
fi

DATA_SUBPATH=$1
SCENE=$(basename "$DATA_SUBPATH")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$REPO_ROOT/data/$DATA_SUBPATH"
HASHNERF="$REPO_ROOT/PocketNeRF"
CONFIG="$HASHNERF/configs/$SCENE.txt"

if [ ! -d "$DATA_DIR" ]; then
  echo "Data folder not found: $DATA_DIR"
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

echo "→ Training BASELINE HashNeRF on scene '$SCENE' (NO structural priors)"
echo "  data = $DATA_DIR"
echo "  config = $CONFIG"
echo

CUDA_VISIBLE_DEVICES=0 \
  python3 "$HASHNERF/run_nerf.py" \
    --config "$CONFIG" \
    --datadir "$DATA_DIR" \
    --finest_res 1024 \
    --expname "${SCENE}_baseline" \
    --use_structural_priors=False \
    --predict_normals=False 
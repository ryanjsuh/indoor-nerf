#!/bin/bash

echo "🧪 Testing PocketNeRF Structural Priors V2 (ManhattanSDF + StructNeRF)"
echo "=================================================================="
echo ""
echo "🔬 Research-based improvements:"
echo "   • Semantic plane detection (floor/wall separation)"
echo "   • Manhattan frame estimation via normal clustering"
echo "   • Spatial-aware normal consistency"
echo "   • Much later activation (iter 3000) for geometry stabilization"
echo "   • 10x smaller weights to prevent overfitting"
echo ""

# Check if scene argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <scene_path>"
    echo "Example: $0 fewshot_iphone/norcliffe_common_room"
    exit 1
fi

SCENE_PATH=$1
SCENE_NAME=$(basename "$SCENE_PATH")

echo "🎯 Target scene: $SCENE_NAME"
echo "📊 Expected improvements:"
echo "   • Train/Test PSNR gap < 6dB (vs current ~13dB)"
echo "   • Non-zero Manhattan/normal consistency losses"
echo "   • Semantic plane detection working"
echo "   • Better generalization"
echo ""

# Set up paths
DATA_DIR="/home/aaronjin/indoor-nerf/data/$SCENE_PATH" 
CONFIG_FILE="/home/aaronjin/indoor-nerf/PocketNeRF/configs/norcliffe_structural_v2.txt"

echo "🏃 Starting training..."
echo "  Data: $DATA_DIR"
echo "  Config: $CONFIG_FILE"
echo ""

# Run training
python run_nerf.py \
    --config "$CONFIG_FILE" \
    --datadir "$DATA_DIR" \
    --expname "${SCENE_NAME}_structural_v2_test"

echo ""
echo "✅ V2 Structural Priors test completed!"
echo "📋 Check logs for:"
echo "   • Semantic detection messages: '🏗️  Semantics: X floor, Y wall points'"
echo "   • Non-zero Manhattan losses broken down by semantic regions"
echo "   • Much smaller train/test PSNR gap"
echo "   • Structural activation at iteration 3000 (much later)" 
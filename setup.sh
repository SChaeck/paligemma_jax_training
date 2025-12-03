#!/bin/bash
# =============================================================================
# PaliGemma Training Environment Setup
# For Vast.ai or similar cloud GPU instances
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PaliGemma Training Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="venv"

# -----------------------------------------------------------------------------
# 1. Python Virtual Environment
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/6] Setting up Python virtual environment...${NC}"

if [ ! -d "$PROJECT_DIR/$VENV_NAME" ]; then
    python3 -m venv "$PROJECT_DIR/$VENV_NAME"
    echo "Created virtual environment: $VENV_NAME"
else
    echo "Virtual environment already exists"
fi

source "$PROJECT_DIR/$VENV_NAME/bin/activate"
pip install --upgrade pip

# -----------------------------------------------------------------------------
# 2. Install Dependencies
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/6] Installing Python dependencies...${NC}"
pip install -r "$PROJECT_DIR/requirements.txt"

# -----------------------------------------------------------------------------
# 3. Clone Big Vision
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/6] Setting up Big Vision...${NC}"

BIG_VISION_PATH="$PROJECT_DIR/../big_vision"

if [ ! -d "$BIG_VISION_PATH" ]; then
    echo "Cloning big_vision repository..."
    git clone --depth 1 https://github.com/google-research/big_vision.git "$BIG_VISION_PATH"
else
    echo "big_vision already exists at $BIG_VISION_PATH"
fi

# -----------------------------------------------------------------------------
# 4. Kaggle Setup & Model Download
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/6] Setting up Kaggle and downloading model...${NC}"

# Check for Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ] && [ -z "$KAGGLE_USERNAME" ]; then
    echo -e "${RED}Warning: Kaggle credentials not found!${NC}"
    echo "Please do one of the following:"
    echo "  1. Place kaggle.json in ~/.kaggle/"
    echo "  2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables"
    echo ""
    echo "To get credentials:"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Click 'Create New Token' under API section"
    echo ""
    read -p "Press Enter to continue without downloading model, or Ctrl+C to exit..."
else
    echo "Downloading PaliGemma model via kagglehub..."
    python3 -c "
import kagglehub
path = kagglehub.model_download('google/paligemma/jax/paligemma-3b-pt-224')
print(f'Model downloaded to: {path}')
"

    # Create symlink to model
    MODEL_CACHE=$(python3 -c "import kagglehub; print(kagglehub.model_download('google/paligemma/jax/paligemma-3b-pt-224'))")
    if [ -f "$MODEL_CACHE/paligemma-3b-pt-224.f16.npz" ]; then
        ln -sf "$MODEL_CACHE/paligemma-3b-pt-224.f16.npz" "$PROJECT_DIR/paligemma-3b-pt-224.f16.npz"
        echo "Created symlink to model checkpoint"
    fi
fi

# -----------------------------------------------------------------------------
# 5. Setup Environment File
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/6] Setting up environment configuration...${NC}"

ENVS_DIR="$PROJECT_DIR/envs"

# Update paths in all env presets
for env_file in "$ENVS_DIR"/.env.*; do
    if [ -f "$env_file" ]; then
        sed -i "s|DATA_BASE_DIR=.*|DATA_BASE_DIR=$PROJECT_DIR/../XVR|g" "$env_file"
        sed -i "s|BIG_VISION_PATH=.*|BIG_VISION_PATH=$BIG_VISION_PATH|g" "$env_file"
    fi
done

echo "Updated paths in envs/*.env files"
echo -e "${YELLOW}Edit envs/.env.* files to set your DATA_BASE_DIR if different${NC}"

# -----------------------------------------------------------------------------
# 6. Verify Installation
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[6/6] Verifying installation...${NC}"

python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    import jax
    devices = jax.devices()
    print(f'JAX devices: {devices}')
    if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
        print('GPU: Available')
    else:
        print('GPU: Not detected (CPU only)')
except Exception as e:
    print(f'JAX: Error - {e}')

try:
    import tensorflow as tf
    print(f'TensorFlow: {tf.__version__}')
except:
    print('TensorFlow: Not installed')

try:
    import flax
    print(f'Flax: {flax.__version__}')
except:
    print('Flax: Not installed')
"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source $VENV_NAME/bin/activate"
echo "  2. Edit .env if needed (especially DATA_BASE_DIR)"
echo "  3. Run overfit test: python scripts/01_overfit_test.py"
echo ""
echo "Available presets (in envs/):"
echo "  - envs/.env.overfit     : Quick overfit test"
echo "  - envs/.env.longrun     : Training with validation"
echo "  - envs/.env.production  : Full production training"
echo ""

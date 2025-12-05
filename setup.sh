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
CONDA_ENV_NAME="paligemma_training"

# GCP paths (update these to your bucket)
GCP_BUCKET="${GCP_BUCKET:-gs://your-bucket-name}"
GCP_CHECKPOINT="${GCP_BUCKET}/checkpoints/pi05_base_paligemma.npz"
GCP_TOKENIZER="${GCP_BUCKET}/assets/paligemma_tokenizer.model"
GCP_XVR_DATA="${GCP_BUCKET}/XVR"

# Local paths
LOCAL_CHECKPOINT="$PROJECT_DIR/checkpoints/pi05_base_paligemma.npz"
LOCAL_TOKENIZER="$PROJECT_DIR/assets/paligemma_tokenizer.model"
LOCAL_XVR_DIR="$PROJECT_DIR/../XVR"
BIG_VISION_PATH="$PROJECT_DIR/../big_vision"

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------
SKIP_DATA=false
SKIP_CHECKPOINT=false
USE_KAGGLE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-checkpoint)
            SKIP_CHECKPOINT=true
            shift
            ;;
        --use-kaggle)
            USE_KAGGLE=true
            shift
            ;;
        --gcp-bucket)
            GCP_BUCKET="$2"
            GCP_CHECKPOINT="${GCP_BUCKET}/checkpoints/pi05_base_paligemma.npz"
            GCP_TOKENIZER="${GCP_BUCKET}/assets/paligemma_tokenizer.model"
            GCP_XVR_DATA="${GCP_BUCKET}/data/XVR"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --gcp-bucket BUCKET   GCP bucket URL (e.g., gs://my-bucket)"
            echo "  --skip-data           Skip XVR data download"
            echo "  --skip-checkpoint     Skip checkpoint download"
            echo "  --use-kaggle          Use Kaggle PaliGemma instead of OpenPI"
            echo ""
            echo "Environment Variables:"
            echo "  GCP_BUCKET            GCP bucket URL (alternative to --gcp-bucket)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# 1. Install Miniconda (if not present)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/6] Checking Miniconda installation...${NC}"

if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    source ~/.bashrc
fi

eval "$(conda shell.bash hook)"
echo "Miniconda: $(which conda)"

# -----------------------------------------------------------------------------
# 2. Create Conda Environment
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/6] Setting up Conda environment...${NC}"

if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi

conda activate "$CONDA_ENV_NAME"
pip install --upgrade pip -q

# -----------------------------------------------------------------------------
# 3. Install Dependencies
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/6] Installing Python dependencies...${NC}"
pip install -r "$PROJECT_DIR/requirements.txt"

# -----------------------------------------------------------------------------
# 4. Clone Big Vision
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/6] Setting up Big Vision...${NC}"

if [ ! -d "$BIG_VISION_PATH" ]; then
    git clone --depth 1 https://github.com/google-research/big_vision.git "$BIG_VISION_PATH"
else
    echo "big_vision already exists"
fi

# -----------------------------------------------------------------------------
# 5. Download Model Files
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/6] Downloading model files...${NC}"

mkdir -p "$PROJECT_DIR/checkpoints" "$PROJECT_DIR/assets"

if [ "$SKIP_CHECKPOINT" = false ]; then
    if [ ! -f "$LOCAL_CHECKPOINT" ]; then
        echo "Downloading checkpoint from GCP..."
        if command -v gsutil &> /dev/null; then
            gsutil cp "$GCP_CHECKPOINT" "$LOCAL_CHECKPOINT"
        else
            echo -e "${RED}gsutil not found. Please install gcloud SDK or download manually:${NC}"
            echo "  $GCP_CHECKPOINT -> $LOCAL_CHECKPOINT"
        fi
    else
        echo "Checkpoint already exists: $LOCAL_CHECKPOINT"
    fi
    
    if [ ! -f "$LOCAL_TOKENIZER" ]; then
        echo "Downloading tokenizer from GCP..."
        if command -v gsutil &> /dev/null; then
            gsutil cp "$GCP_TOKENIZER" "$LOCAL_TOKENIZER"
        else
            echo -e "${RED}gsutil not found. Please download manually:${NC}"
            echo "  $GCP_TOKENIZER -> $LOCAL_TOKENIZER"
        fi
    else
        echo "Tokenizer already exists: $LOCAL_TOKENIZER"
    fi
fi

# -----------------------------------------------------------------------------
# 6. Download XVR Dataset
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[6/6] Setting up XVR dataset...${NC}"

if [ "$SKIP_DATA" = false ]; then
    if [ ! -f "$LOCAL_XVR_DIR/train.jsonl" ]; then
        echo "Downloading XVR dataset from GCP..."
        mkdir -p "$LOCAL_XVR_DIR"
        if command -v gsutil &> /dev/null; then
            gsutil -m rsync -r "$GCP_XVR_DATA" "$LOCAL_XVR_DIR"
        else
            echo -e "${RED}gsutil not found. Please download manually to: $LOCAL_XVR_DIR${NC}"
        fi
    else
        echo "XVR dataset already exists: $LOCAL_XVR_DIR"
    fi
fi

# -----------------------------------------------------------------------------
# Update Environment Files
# -----------------------------------------------------------------------------
for env_file in "$PROJECT_DIR"/envs/.env.*; do
    if [ -f "$env_file" ]; then
        sed -i "s|DATA_BASE_DIR=.*|DATA_BASE_DIR=$LOCAL_XVR_DIR|g" "$env_file"
        sed -i "s|BIG_VISION_PATH=.*|BIG_VISION_PATH=$BIG_VISION_PATH|g" "$env_file"
    fi
done

# -----------------------------------------------------------------------------
# Verify Installation
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Verifying installation...${NC}"

python3 -c "
import jax
print(f'JAX: {jax.__version__}')
print(f'Devices: {jax.devices()}')
import flax
print(f'Flax: {flax.__version__}')
"

echo ""
[ -f "$LOCAL_CHECKPOINT" ] && echo -e "${GREEN}✓ Checkpoint: $LOCAL_CHECKPOINT${NC}" || echo -e "${RED}✗ Checkpoint missing${NC}"
[ -f "$LOCAL_TOKENIZER" ] && echo -e "${GREEN}✓ Tokenizer: $LOCAL_TOKENIZER${NC}" || echo -e "${RED}✗ Tokenizer missing${NC}"
[ -f "$LOCAL_XVR_DIR/train.jsonl" ] && echo -e "${GREEN}✓ XVR Data: $LOCAL_XVR_DIR${NC}" || echo -e "${RED}✗ XVR Data missing${NC}"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "  conda activate $CONDA_ENV_NAME"
echo "  python scripts/01_overfit_test.py  # Quick test"
echo "  python scripts/03_train_production.py --env envs/.env.openpi  # Full training"
echo ""

# PaliGemma Fine-tuning

PaliGemma fine-tuning on XVR dataset with JAX.

## Structure

```
config_pi_training/
├── envs/                 # Environment presets
│   ├── .env.example      # Config template
│   ├── .env.overfit      # Overfit test preset
│   ├── .env.longrun      # Training with validation curves
│   ├── .env.production   # Production training preset
│   └── .env.openpi       # OpenPI training preset
│
├── src/                  # Core library
│   ├── config.py         # Config from environment
│   ├── model.py          # Model loading
│   ├── data.py           # Dataset & preprocessing
│   ├── training.py       # Training step
│   └── evaluation.py     # Evaluation
│
├── scripts/              # Entry points
│   ├── 01_overfit_test.py      # Step 1: Verify training works
│   ├── 02_train_validate.py    # Step 2: Train with curves
│   ├── 03_train_production.py  # Step 3: Final training
│   ├── evaluate.py             # Standalone evaluation
│   └── plot_curves.py          # Visualize curves
│
├── setup.sh              # Auto setup for cloud instances
└── requirements.txt
```

## Quick Start (Vast.ai / Cloud GPU)

```bash
# 1. Clone the repo
git clone https://github.com/SChaeck/paligemma_jax_training.git
cd paligemma_jax_training

# 2. Set up Kaggle credentials (for model download)
mkdir -p ~/.kaggle
echo '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 3. Run setup script (installs deps, downloads model, clones big_vision)
chmod +x setup.sh
./setup.sh

# 4. Activate environment and prepare data
source venv/bin/activate
# Copy your XVR dataset or update DATA_BASE_DIR in envs/.env.*

# 5. Run training
python scripts/01_overfit_test.py
```

## Manual Setup

```bash
pip install -r requirements.txt
# Edit envs/.env.* files to set DATA_BASE_DIR and other paths
```

## Usage

### Step 1: Verify Training (Overfit Test)

Small dataset, expect ~100% accuracy:

```bash
python scripts/01_overfit_test.py
```

### Step 2: Train with Validation Curves

Full training, monitor for overfitting:

```bash
python scripts/02_train_validate.py
python scripts/plot_curves.py outputs/longrun/training_curves.json
```

### Step 3: Production Training

Final training with tuned hyperparameters:

```bash
python scripts/03_train_production.py
```

### Custom Config

```bash
python scripts/02_train_validate.py --env envs/.env.custom
```

## Key Parameters (envs/.env.*)

```bash
# Training
LEARNING_RATE=0.03
BATCH_SIZE=8
NUM_EPOCHS=10
TRAINABLE_PARAMS=attention_only  # attention_only | full_llm | full_model

# Data
DATA_BASE_DIR=/path/to/XVR
MAX_TRAIN_SAMPLES=         # Empty = all

# Logging
LOG_EVERY=10
EVAL_EVERY=100
USE_WANDB=false
```

## Requirements

- Python 3.10+
- JAX with CUDA
- NVIDIA GPU (tested on L40S 46GB)

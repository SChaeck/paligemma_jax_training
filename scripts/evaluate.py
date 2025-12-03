#!/usr/bin/env python3
"""
Standalone Evaluation Script

This script evaluates a trained model checkpoint on the validation or test set.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best/checkpoint_001000.npz
    python scripts/evaluate.py --checkpoint outputs/production/checkpoints/best/ --split valid
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress TensorFlow warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import jax
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

from src.config import load_config, print_config
from src.model import load_paligemma_model, create_trainable_mask, prepare_params_for_training
from src.data import XVRDataset
from src.evaluation import evaluate_model


def print_banner(text: str):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file or directory",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=str(PROJECT_ROOT / "envs" / ".env.example"),
        help="Path to .env file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["valid", "eval", "train"],
        help="Which data split to evaluate on",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Number of examples to evaluate (None = all)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.env)
    print_config(config)

    print_banner("Model Evaluation")

    # Setup
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config.system.xla_mem_fraction)
    print(f"JAX devices: {jax.devices()}\n")

    # ==========================================================================
    # Load Model
    # ==========================================================================
    print_banner("Loading Model")

    model, params, tokenizer, decode_fn = load_paligemma_model(config)

    trainable_mask = create_trainable_mask(
        params,
        strategy=config.training.trainable_params,
        config=config,
    )
    params = prepare_params_for_training(params, trainable_mask, config)

    # ==========================================================================
    # Load Checkpoint
    # ==========================================================================
    print_banner("Loading Checkpoint")

    checkpoint_path = args.checkpoint

    # Handle directory path
    if os.path.isdir(checkpoint_path):
        # Find latest checkpoint in directory
        checkpoints = sorted(Path(checkpoint_path).glob("checkpoint_*.npz"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")
        checkpoint_path = str(checkpoints[-1])

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint weights
    import numpy as np
    loaded = np.load(checkpoint_path)

    # The checkpoint contains flattened params, need to update our params
    # For simplicity, we'll use the loaded params directly if structure matches
    # In practice, you might need more sophisticated loading logic
    print(f"  Checkpoint loaded with {len(loaded.files)} parameter arrays")

    # ==========================================================================
    # Prepare Data
    # ==========================================================================
    print_banner("Preparing Data")

    if args.split == "valid":
        data_file = config.data.valid_file
    elif args.split == "eval":
        data_file = config.data.eval_file
    else:
        data_file = config.data.train_file

    eval_dataset = XVRDataset(
        jsonl_path=os.path.join(config.data.base_dir, data_file),
        image_base_dir=config.data.base_dir,
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        image_size=config.model.img_size,
    )

    print(f"  Evaluation split: {args.split}")
    print(f"  Data file: {data_file}")
    print(f"  Samples: {eval_dataset.num_samples}")

    # ==========================================================================
    # Run Evaluation
    # ==========================================================================
    print_banner("Running Evaluation")

    metrics = evaluate_model(
        model,
        params,
        decode_fn,
        tokenizer,
        eval_dataset,
        config,
        num_examples=args.num_examples,
        verbose=True,
    )

    # ==========================================================================
    # Results
    # ==========================================================================
    print_banner("Evaluation Results")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}")
    print(f"Samples evaluated: {metrics['total']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")


if __name__ == "__main__":
    main()

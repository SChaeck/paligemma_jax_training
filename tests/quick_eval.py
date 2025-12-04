#!/usr/bin/env python3
"""Quick eval script for comparing vanilla vs pi0.5_base on XVR."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, 
                        choices=['vanilla', 'pi05'],
                        help='Which checkpoint to evaluate')
    parser.add_argument('--num-examples', type=int, default=50)
    args = parser.parse_args()
    
    # Set paths
    if args.checkpoint == 'vanilla':
        # Kaggle PaliGemma (download via kagglehub if needed)
        import kagglehub
        path = kagglehub.model_download('google/paligemma/jax/paligemma-3b-pt-224')
        ckpt_path = str(Path(path) / "paligemma-3b-pt-224.f16.npz")
        name = "Vanilla PaliGemma"
    else:
        # OpenPI pi0.5_base checkpoint
        ckpt_path = str(PROJECT_ROOT / "checkpoints" / "pi05_base_paligemma.npz")
        name = "pi0.5_base"
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Examples: {args.num_examples}")
    print(f"{'='*60}\n")
    
    # Check file exists
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found: {ckpt_path}")
        return
    
    import jax
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.set_visible_devices([], "TPU")
    
    from src.config import load_config
    from src.model import load_paligemma_model, create_trainable_mask, prepare_params_for_training
    from src.data import XVRDataset
    from src.evaluation import evaluate_model
    
    # Load config
    env_file = str(PROJECT_ROOT / "envs" / ".env.example")
    config = load_config(env_file)
    
    # Fix data path
    config.data.base_dir = "/home/suchae/pi_workspace/XVR"
    
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config.system.xla_mem_fraction)
    print(f"JAX devices: {jax.devices()}")
    
    # Override checkpoint path
    config.model.checkpoint_path = ckpt_path
    config.model.checkpoint_url = None
    
    # Check eval file
    eval_file = os.path.join(config.data.base_dir, config.data.eval_file)
    if not os.path.exists(eval_file):
        print(f"Warning: Eval file not found: {eval_file}")
        print("Using valid.jsonl instead...")
        config.data.eval_file = "valid.jsonl"
        eval_file = os.path.join(config.data.base_dir, config.data.eval_file)
    
    print(f"Eval file: {eval_file}")
    
    # Load model
    print("\nLoading model...")
    model, params, tokenizer, decode_fn = load_paligemma_model(config)
    
    # Prepare params
    trainable_mask = create_trainable_mask(params, strategy="attention_only", config=config)
    params = prepare_params_for_training(params, trainable_mask, config)
    
    # Load eval dataset
    eval_dataset = XVRDataset(
        jsonl_path=eval_file,
        image_base_dir=config.data.base_dir,
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        image_size=config.model.img_size,
    )
    
    print(f"\nEvaluating on {args.num_examples} examples...")
    
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
    
    print(f"\n{'='*60}")
    print(f"RESULT: {name}")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


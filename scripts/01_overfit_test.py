#!/usr/bin/env python3
"""
Overfitting Test Script

This script verifies that the training pipeline works correctly by overfitting
on a small subset of data. Expected behavior:
- Training loss should decrease to near zero
- Training accuracy should reach near 100%

Usage:
    python scripts/01_overfit_test.py
    python scripts/01_overfit_test.py --env .env.overfit
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress TensorFlow warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force single GPU mode to avoid NCCL/sharding issues
# Must be set BEFORE importing JAX
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

from src.config import load_config, validate_config, print_config
from src.model import (
    load_paligemma_model,
    create_trainable_mask,
    prepare_params_for_training,
    save_checkpoint,
)
from src.data import XVRDataset, create_train_iterator, collate_batch
from src.training import (
    create_learning_rate_schedule,
    compiled_train_step,
    compiled_train_step_adam,
    compiled_train_step_with_accumulation,
    create_optimizer,
    create_optimizer_state,
    MetricsTracker,
    create_data_sharding,
    shard_batch,
)
from src.evaluation import evaluate_model


def print_banner(text: str):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def setup_environment(config):
    """Setup environment variables and JAX configuration."""
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config.system.xla_mem_fraction)

    if config.system.tf_allow_growth:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Set random seed for reproducibility
    rng_key = jax.random.PRNGKey(config.training.seed)
    print(f"Random seed:  {config.training.seed}")

    # Set precision
    if config.training.precision == "bfloat16":
        jax.config.update("jax_default_matmul_precision", "bfloat16")
        print(f"Precision:    bfloat16")
    elif config.training.precision == "float16":
        jax.config.update("jax_default_matmul_precision", "float16")
        print(f"Precision:    float16")
    else:
        jax.config.update("jax_default_matmul_precision", "float32")
        print(f"Precision:    float32")

    print(f"JAX version:  {jax.__version__}")
    print(f"JAX devices:  {jax.device_count()}")
    print(f"Device list:  {jax.devices()}\n")
    
    return rng_key


def run_overfit_test(config):
    """
    Run overfitting test to verify training works.

    Args:
        config: Training configuration
    """
    print_banner("Overfitting Test")
    print("This test verifies the training pipeline by overfitting on a small dataset.")
    print("Expected: Training loss -> 0, Training accuracy -> 100%\n")

    # Setup
    rng_key = setup_environment(config)

    # Create output directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Load Model
    # ==========================================================================
    print_banner("Step 1: Loading Model")

    model, params, tokenizer, decode_fn = load_paligemma_model(config)

    # Create trainable mask and prepare params
    trainable_mask = create_trainable_mask(
        params,
        strategy=config.training.trainable_params,
        config=config,
    )
    params = prepare_params_for_training(params, trainable_mask, config)

    # ==========================================================================
    # Prepare Data (Small Subset)
    # ==========================================================================
    print_banner("Step 2: Preparing Data (Small Subset)")

    train_dataset = XVRDataset(
        jsonl_path=os.path.join(config.data.base_dir, config.data.train_file),
        image_base_dir=config.data.base_dir,
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        image_size=config.model.img_size,
        max_samples=config.data.max_train_samples,
        shuffle_buffer_size=config.data.shuffle_buffer_size,
    )

    # For overfit test, use same data for train and eval
    eval_dataset = XVRDataset(
        jsonl_path=os.path.join(config.data.base_dir, config.data.train_file),
        image_base_dir=config.data.base_dir,
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        image_size=config.model.img_size,
        max_samples=config.data.max_train_samples,
        shuffle_buffer_size=config.data.shuffle_buffer_size,
    )

    print(f"  Training samples: {train_dataset.num_samples}")
    print(f"  Eval samples (same as train): {eval_dataset.num_samples}")

    train_iterator = create_train_iterator(
        train_dataset,
        batch_size=config.training.batch_size,
        prompt_prefix=config.data.prompt_prefix,
        max_images=config.training.max_images,
    )

    # ==========================================================================
    # Setup Training
    # ==========================================================================
    print_banner("Step 3: Setting Up Training")

    steps_per_epoch = train_dataset.num_samples // config.training.batch_size
    total_steps = steps_per_epoch * config.training.num_epochs

    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Optimizer: AdamW")

    # Create AdamW optimizer with warmup + cosine decay (matches paligemma2 PyTorch)
    optimizer = create_optimizer(
        learning_rate=config.training.learning_rate,
        total_steps=total_steps,
        warmup_percent=config.training.warmup_percent,
        weight_decay=0.0,  # No weight decay for fine-tuning
        max_grad_norm=config.training.max_grad_norm,
    )
    opt_state = create_optimizer_state(optimizer, params)
    print(f"  Optimizer state initialized")

    # Keep lr_schedule for logging purposes only
    lr_schedule = create_learning_rate_schedule(
        base_learning_rate=config.training.learning_rate,
        total_steps=total_steps,
        warmup_percent=config.training.warmup_percent,
        schedule_type=config.training.lr_schedule,
        config=config,
    )

    data_sharding = create_data_sharding(config)
    metrics_tracker = MetricsTracker()

    # Track metrics for plotting
    train_losses = []
    train_accuracies = []
    eval_steps = []

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    print_banner("Step 4: Overfitting Training")

    print("Starting training loop...\n")
    print("Watch for:")
    print("  - Loss should decrease steadily")
    print("  - Accuracy on training data should approach 100%\n")

    start_time = time.time()
    accum_steps = config.training.gradient_accumulation_steps

    for step in range(1, total_steps + 1):
        step_start = time.time()

        # Gradient accumulation: collect multiple batches
        accumulated_loss = 0.0
        accumulated_grads = None

        for accum_idx in range(accum_steps):
            # Get batch - collect batch_size samples and use proper collation
            batch_samples = [next(train_iterator) for _ in range(config.training.batch_size)]
            batch = collate_batch(
                batch_samples,
                max_images=config.training.max_images,
                image_size=config.model.img_size,
            )
            batch = shard_batch(batch, data_sharding, config)

            if accum_steps == 1:
                # No accumulation - use Adam optimizer
                lr = lr_schedule(step)  # for logging only
                params, opt_state, loss = compiled_train_step_adam(
                    params,
                    opt_state,
                    batch,
                    model,
                    optimizer,
                    trainable_mask,
                )
                accumulated_loss = loss
            else:
                # Accumulate gradients manually
                from src.training import compute_loss_and_grads
                loss, grads = compute_loss_and_grads(params, batch, model)
                accumulated_loss += loss

                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = jax.tree.map(lambda x, y: x + y, accumulated_grads, grads)

        # Apply accumulated gradients
        if accum_steps > 1:
            lr = lr_schedule(step)
            # Average gradients
            accumulated_grads = jax.tree.map(lambda g: g / accum_steps, accumulated_grads)
            accumulated_loss = accumulated_loss / accum_steps

            # Apply gradient clipping and update
            from src.training import apply_gradients
            params = apply_gradients(params, accumulated_grads, trainable_mask, lr, config.training.max_grad_norm)
            loss = accumulated_loss

        loss = jax.device_get(loss)
        step_time = time.time() - step_start

        train_losses.append(float(loss))

        # Log progress
        if step % config.logging.log_every == 0:
            print(
                f"Step {step:4d}/{total_steps} | "
                f"Loss: {loss:.4f} | "
                f"LR: {lr:.6f} | "
                f"Time: {step_time:.2f}s"
            )

        # Evaluate on training data
        if step % config.logging.eval_every == 0:
            eval_metrics = evaluate_model(
                model, params, decode_fn, tokenizer,
                eval_dataset, config,
                num_examples=config.eval.num_examples,
                verbose=True,
            )
            train_accuracies.append(eval_metrics['accuracy'])
            eval_steps.append(step)

            print(f"\n  Training Accuracy: {eval_metrics['accuracy']:.2%}")

            # Check if overfit succeeded
            if eval_metrics['accuracy'] >= 0.95:
                print("\n" + "=" * 80)
                print("SUCCESS: Model achieved >= 95% accuracy on training data!")
                print("The training pipeline is working correctly.")
                print("=" * 80)

    # ==========================================================================
    # Final Evaluation
    # ==========================================================================
    print_banner("Step 5: Final Evaluation")

    # Save debug results for final evaluation
    final_results_path = os.path.join(config.checkpoint_dir, "overfit_eval_results.json")

    final_metrics = evaluate_model(
        model, params, decode_fn, tokenizer,
        eval_dataset, config,
        num_examples=None,  # Evaluate all
        verbose=True,
        save_results_path=final_results_path,
        max_saved_samples=100,
    )

    # ==========================================================================
    # Summary
    # ==========================================================================
    total_time = time.time() - start_time
    print_banner("Overfit Test Summary")

    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Final loss: {train_losses[-1]:.4f}")
    print(f"Final training accuracy: {final_metrics['accuracy']:.2%}")

    if final_metrics['accuracy'] >= 0.90:
        print("\n[PASS] Overfitting test PASSED!")
        print("The model successfully memorized the training data.")
        print("You can proceed to full training with train/validation split.")
    else:
        print("\n[WARN] Overfitting test did not reach 90% accuracy.")
        print("Consider:")
        print("  - Increasing learning rate")
        print("  - Training for more epochs")
        print("  - Checking data preprocessing")

    # Save final checkpoint
    save_checkpoint(
        params,
        total_steps,
        config.checkpoint_dir,
        config,
        keep_last_n=1,
    )

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Run overfitting test")
    parser.add_argument(
        "--env",
        type=str,
        default=str(PROJECT_ROOT / "envs" / ".env.overfit"),
        help="Path to .env file",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.env)
    validate_config(config)
    print_config(config)

    try:
        run_overfit_test(config)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

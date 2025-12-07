#!/usr/bin/env python3
"""
Production Training Script

This script runs final production training with optimized hyperparameters.
Use after validating the pipeline with overfit test and long training experiments.

Usage:
    python scripts/03_train_production.py
    python scripts/03_train_production.py --env .env.production
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress TensorFlow warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Optional: Force single GPU mode if CUDA_VISIBLE_DEVICES is not already set
# This can be useful to avoid NCCL/sharding issues, but prevents multi-GPU training
# To use multi-GPU: either don't set this env var, or set it to multiple GPUs (e.g., '0,1,2,3')
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    # Default behavior: use all available GPUs
    # To force single GPU, uncomment the line below:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    pass

import jax
import jax.numpy as jnp
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
    compiled_train_step_adam,
    compute_loss_and_grads_for_accum,
    apply_accumulated_gradients_adam,
    create_optimizer,
    create_optimizer_state,
    MetricsTracker,
    create_data_sharding,
    shard_batch,
    # Multi-GPU support
    setup_bfloat16,
    print_device_info,
    get_num_devices,
    create_pmap_train_step_adam,
    replicate_params,
    unreplicate_params,
    shard_batch_for_pmap,
)
from src.evaluation import evaluate_model


def print_banner(text: str):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def save_training_batch(
    batch: Dict,
    batch_samples: list,
    step: int,
    loss: float,
    decode_fn: Any,
    tokenizer: Any,
    output_dir: str,
    max_samples: int = 4,
):
    """
    Save training batch samples for debugging - EXACTLY like validation does it.

    Simply saves the original text from batch_samples (input_prompt and ground_truth).
    """
    import PIL.Image

    # Create output directory
    batch_dir = os.path.join(output_dir, f"train_batch_step_{step:06d}")
    os.makedirs(batch_dir, exist_ok=True)
    images_dir = os.path.join(batch_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Unshard batch if needed
    if batch['image'].ndim == 6:  # Sharded for pmap
        batch_images = np.array(batch['image'][0])
    else:
        batch_images = np.array(batch['image'])

    # Limit to max_samples
    num_samples = min(len(batch_images), max_samples, len(batch_samples))

    samples_info = []

    for i in range(num_samples):
        # Get sample metadata - THIS IS THE KEY! Just like validation does it
        sample = batch_samples[i]
        sample_id = sample.get('sample_id', f'sample_{i}')
        input_prompt = sample.get('input_prompt', '')
        ground_truth = sample.get('ground_truth', '')
        num_images = sample.get('num_images', 0)

        sample_info = {
            "index": i,
            "sample_id": sample_id,
            "input_prompt": input_prompt,  # Original text - NOT decoded!
            "ground_truth": ground_truth,  # Original answer - NOT decoded!
            "num_images": num_images,
            "step": step,
            "loss": float(loss),
        }

        # Save images
        images = batch_images[i]
        image_paths = []
        for img_idx in range(num_images):
            img = images[img_idx]

            # Convert to [0, 255]
            if img.min() < 0:
                img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            elif img.max() <= 1.0:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)

            img_path = os.path.join(images_dir, f"sample_{i:02d}_image_{img_idx}.png")
            PIL.Image.fromarray(img).save(img_path)
            image_paths.append(f"images/sample_{i:02d}_image_{img_idx}.png")

        sample_info["image_paths"] = image_paths
        samples_info.append(sample_info)

    # Save JSON summary - simple format like validation
    summary = {
        "step": step,
        "loss": float(loss),
        "num_samples_saved": num_samples,
        "samples": samples_info,
    }

    summary_path = os.path.join(batch_dir, "batch_info.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Saved training batch to: {batch_dir}")


def setup_environment(config):
    """Setup environment variables and JAX configuration."""
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config.system.xla_mem_fraction)

    if config.system.tf_allow_growth:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Set random seed for reproducibility
    rng_key = jax.random.PRNGKey(config.training.seed)
    print(f"Random seed:  {config.training.seed}")

    # Set precision - use setup_bfloat16 for bfloat16 (matches big_vision/OpenPi)
    if config.training.precision == "bfloat16":
        setup_bfloat16()
        print(f"Precision:    bfloat16")
    elif config.training.precision == "float16":
        jax.config.update("jax_default_matmul_precision", "float16")
        print(f"Precision:    float16")
    else:
        jax.config.update("jax_default_matmul_precision", "float32")
        print(f"Precision:    float32")

    print(f"JAX version:  {jax.__version__}")
    print_device_info()
    
    return rng_key


def setup_wandb(config):
    """Initialize Weights & Biases if enabled. Tries multiple methods to ensure success."""
    if not config.logging.use_wandb:
        return None

    try:
        import wandb
        import wandb.apis.public as wandb_api
    except ImportError:
        print("  W&B not installed, skipping")
        return None

    base_kwargs = {
        "project": config.logging.wandb_project,
        "name": f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "config": {
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.batch_size,
            "num_epochs": config.training.num_epochs,
            "trainable_params": config.training.trainable_params,
            "lr_schedule": config.training.lr_schedule,
        },
        "tags": ["production"],
        # Short timeout to avoid blocking - will fallback to offline
        "settings": wandb.Settings(init_timeout=10),
    }

    # Get current username first
    current_user = None
    try:
        api = wandb_api.Api()
        # viewer is a property, not a method
        viewer = api.viewer
        if hasattr(viewer, 'username'):
            current_user = viewer.username
        elif hasattr(viewer, 'entity'):
            current_user = viewer.entity
        elif isinstance(viewer, dict):
            current_user = viewer.get("username") or viewer.get("entity")
    except Exception as e:
        # Try alternative method
        try:
            import subprocess
            result = subprocess.run(
                ["wandb", "whoami"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse output like "schaeck (configint)"
                output = result.stdout.strip()
                current_user = output.split()[0] if output else None
        except Exception:
            pass
        if not current_user:
            print(f"  Could not get current user: {e}")

    # Try offline mode FIRST to avoid blocking
    # This ensures training can start immediately
    offline_kwargs = {
        "project": config.logging.wandb_project,
        "name": f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "mode": "offline",
        "config": {
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.batch_size,
            "num_epochs": config.training.num_epochs,
            "trainable_params": config.training.trainable_params,
            "lr_schedule": config.training.lr_schedule,
        },
        "tags": ["production"],
    }
    if current_user:
        offline_kwargs["entity"] = current_user
    elif config.logging.wandb_entity:
        offline_kwargs["entity"] = config.logging.wandb_entity
    
    # Try offline mode first (fast, no blocking)
    try:
        print("  Initializing W&B in offline mode (fast, no blocking)...")
        wandb.init(**offline_kwargs)
        if wandb.run is not None:
            print("  W&B initialized in OFFLINE mode")
            print("  Note: Run 'wandb sync' after training to upload logs")
            return wandb
    except Exception as e:
        print(f"  Offline mode failed: {e}")
    
    # If offline fails, try online with short timeout
    print("  Trying online mode with short timeout (10s)...")
    methods = []
    
    # Method 1: Use explicitly configured entity
    if config.logging.wandb_entity:
        methods.append(("explicit entity", {**base_kwargs, "entity": config.logging.wandb_entity}))
    
    # Method 2: Use current user directly
    if current_user:
        methods.append(("current user", {**base_kwargs, "entity": current_user}))

    # Try each method until one succeeds
    last_error = None
    import os
    import signal
    
    for method_name, init_kwargs in methods:
        try:
            # Completely reset wandb state before each attempt
            try:
                if wandb.run is not None:
                    wandb.finish()
            except:
                pass
            # Clear any pending wandb state by resetting the run
            try:
                wandb.run = None
            except:
                pass
            
            # Set environment variable BEFORE wandb.init() if entity is specified
            old_entity_env = None
            if "entity" in init_kwargs and init_kwargs["entity"]:
                old_entity_env = os.environ.get("WANDB_ENTITY")
                os.environ["WANDB_ENTITY"] = init_kwargs["entity"]
            
            try:
                # Add reinit to ensure clean state
                init_kwargs_final = {**init_kwargs, "reinit": True}
                
                # Call wandb.init() - this may block, but we need to try
                print(f"  Trying wandb.init() with method: {method_name}...")
                wandb.init(**init_kwargs_final)
                
                # Verify initialization succeeded
                if wandb.run is not None:
                    # Double-check entity is correct
                    if "entity" in init_kwargs and wandb.run.entity != init_kwargs["entity"]:
                        print(f"  Warning: Entity mismatch. Expected {init_kwargs['entity']}, got {wandb.run.entity}")
                    print(f"  W&B initialized (method: {method_name})")
                    print(f"  W&B entity: {wandb.run.entity}")
                    print(f"  W&B run URL: {wandb.run.url}")
                    return wandb
                else:
                    raise RuntimeError("wandb.init() returned but wandb.run is None")
            finally:
                # Restore environment variable
                if old_entity_env is not None:
                    if old_entity_env:
                        os.environ["WANDB_ENTITY"] = old_entity_env
                    elif "WANDB_ENTITY" in os.environ:
                        del os.environ["WANDB_ENTITY"]
        except Exception as e:
            last_error = e
            try:
                if wandb.run is not None:
                    wandb.finish()
            except:
                pass
            continue
    
    # All online methods failed - offline should have worked, but if not, continue
    print(f"  WARNING: Failed to initialize W&B online after trying {len(methods)} methods")
    print(f"  Last error: {last_error}")
    print("  Continuing without W&B logging...")
    return None


def train_production(config):
    """
    Run production training.

    Args:
        config: Training configuration
    """
    print_banner("Production Training")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Started at: {timestamp}")
    print(f"Experiment: {config.experiment_name}\n")

    # Setup
    rng_key = setup_environment(config)
    wandb = setup_wandb(config)

    # Create output directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config_save_path = os.path.join(config.logging.output_dir, f"config_{timestamp}.json")
    config_dict = {
        "experiment_name": config.experiment_name,
        "timestamp": timestamp,
        "model": config.model.__dict__,
        "training": config.training.__dict__,
        "data": config.data.__dict__,
        "logging": config.logging.__dict__,
        "eval": config.eval.__dict__,
        "system": config.system.__dict__,
    }
    Path(config.logging.output_dir).mkdir(parents=True, exist_ok=True)
    with open(config_save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to: {config_save_path}\n")

    # ==========================================================================
    # Load Model
    # ==========================================================================
    print_banner("Step 1: Loading Model")

    model, params, tokenizer, decode_fn = load_paligemma_model(config)

    trainable_mask = create_trainable_mask(
        params,
        strategy=config.training.trainable_params,
        config=config,
    )
    params = prepare_params_for_training(params, trainable_mask, config)

    # ==========================================================================
    # Prepare Data
    # ==========================================================================
    print_banner("Step 2: Preparing Data")

    train_dataset = XVRDataset(
        jsonl_path=os.path.join(config.data.base_dir, config.data.train_file),
        image_base_dir=config.data.base_dir,
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        image_size=config.model.img_size,
        max_samples=config.data.max_train_samples,
        shuffle_buffer_size=config.data.shuffle_buffer_size,
    )

    valid_dataset = XVRDataset(
        jsonl_path=os.path.join(config.data.base_dir, config.data.valid_file),
        image_base_dir=config.data.base_dir,
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        image_size=config.model.img_size,
        max_samples=config.data.max_eval_samples,
        shuffle_buffer_size=config.data.shuffle_buffer_size,
    )

    print(f"  Training samples: {train_dataset.num_samples}")
    print(f"  Validation samples: {valid_dataset.num_samples}")

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

    # Account for gradient accumulation in step calculation
    effective_batch_size = config.training.batch_size * config.training.gradient_accumulation_steps
    steps_per_epoch = train_dataset.num_samples // effective_batch_size
    total_steps = steps_per_epoch * config.training.num_epochs

    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Optimizer: AdamW")
    print(f"  Estimated time: ~{total_steps * 0.25 / 60:.1f} minutes")

    # Create AdamW optimizer with warmup + cosine decay (matches paligemma2 PyTorch)
    optimizer = create_optimizer(
        learning_rate=config.training.learning_rate,
        total_steps=total_steps,
        warmup_percent=config.training.warmup_percent,
        weight_decay=0.0,  # No weight decay for fine-tuning
        max_grad_norm=config.training.max_grad_norm,
    )

    # Keep lr_schedule for logging purposes only
    lr_schedule = create_learning_rate_schedule(
        base_learning_rate=config.training.learning_rate,
        total_steps=total_steps,
        warmup_percent=config.training.warmup_percent,
        schedule_type=config.training.lr_schedule,
        config=config,
    )

    metrics_tracker = MetricsTracker()

    best_accuracy = 0.0
    best_step = 0

    # ==========================================================================
    # Multi-GPU Setup
    # ==========================================================================
    num_devices = get_num_devices()
    use_pmap = config.training.use_pmap and num_devices > 1

    if use_pmap:
        print(f"\n[Multi-GPU] Using pmap with {num_devices} devices")
        print(f"[Multi-GPU] Effective batch size per step: {config.training.batch_size} (split across {num_devices} GPUs)")

        # For pmap, params must be on CPU before replication
        # Initialize optimizer state on CPU
        opt_state = create_optimizer_state(optimizer, params)
        print(f"  Optimizer state initialized on CPU")

        # Create pmap training step with Adam optimizer
        pmap_train_step = create_pmap_train_step_adam(model, optimizer)

        # Replicate params, opt_state, and mask across devices
        print(f"  Replicating params across {num_devices} devices...")
        params = replicate_params(params)
        opt_state = replicate_params(opt_state)
        trainable_mask = replicate_params(trainable_mask)
        print(f"  Replication complete")
    else:
        print(f"\n[Single-GPU] Using jit compiled training with Adam optimizer")

        # Move params to single GPU (avoid device mismatch)
        first_device = jax.devices()[0]
        params = jax.device_put(params, first_device)
        trainable_mask = jax.device_put(trainable_mask, first_device)
        print(f"  Moved params to single GPU: {first_device}")

        # Initialize optimizer state after moving params
        opt_state = create_optimizer_state(optimizer, params)
        print(f"  Optimizer state initialized")

        data_sharding = create_data_sharding(config)

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    print_banner("Step 4: Training")

    print("Starting production training...\n")
    start_time = time.time()

    # Gradient accumulation: collect multiple batches before updating
    accum_steps = config.training.gradient_accumulation_steps
    accum_batches = []
    effective_step = 0

    for step in range(1, total_steps + 1):
        step_start = time.time()

        # Get batch - collect batch_size samples and use proper collation
        # This handles variable number of images per sample correctly
        batch_samples = [next(train_iterator) for _ in range(config.training.batch_size)]
        batch = collate_batch(
            batch_samples,
            max_images=config.training.max_images,
            image_size=config.model.img_size,
        )

        # Shard batch based on single/multi GPU mode
        if use_pmap:
            batch = shard_batch_for_pmap(batch, num_devices)
        else:
            batch = shard_batch(batch, data_sharding, config)

        accum_batches.append(batch)

        # Only update when we have accumulated enough batches
        should_update = len(accum_batches) >= accum_steps or step == total_steps
        
        if should_update:
            effective_step += 1
            # Get learning rate for logging (optimizer handles schedule internally)
            lr = lr_schedule(effective_step)

            # Save training batch BEFORE training (every log_every steps)
            # This shows exactly what the model sees as input
            if step % config.logging.log_every == 0:
                try:
                    save_training_batch(
                        batch=accum_batches[0],  # Save first accumulated batch
                        batch_samples=batch_samples,  # Original samples with metadata
                        step=step,
                        loss=0.0,  # Will be updated after training
                        decode_fn=decode_fn,
                        tokenizer=tokenizer,
                        output_dir=config.checkpoint_dir,
                        max_samples=4,
                    )
                except Exception as e:
                    print(f"  Warning: Failed to save training batch: {e}")

            # Training step with Adam optimizer
            if use_pmap:
                # Multi-GPU with pmap
                if accum_steps == 1:
                    # Direct pmap update (no gradient accumulation)
                    params, opt_state, loss = pmap_train_step(
                        params, opt_state, accum_batches[0], trainable_mask
                    )
                    # Loss is same on all devices, take first
                    loss = loss[0]
                else:
                    # Gradient accumulation with pmap - NOT RECOMMENDED
                    # For simplicity with pmap, use gradient_accumulation_steps=1
                    raise NotImplementedError(
                        "Gradient accumulation with pmap is not yet supported. "
                        "Please set GRADIENT_ACCUMULATION_STEPS=1 or USE_PMAP=false"
                    )
            else:
                # Single-GPU path
                if accum_steps > 1 and len(accum_batches) > 1:
                    # Gradient accumulation with Adam
                    accumulated_grads = None
                    accumulated_loss = 0.0

                    for batch in accum_batches:
                        loss, grads = compute_loss_and_grads_for_accum(params, batch, model, trainable_mask)
                        accumulated_loss += loss
                        if accumulated_grads is None:
                            accumulated_grads = grads
                        else:
                            accumulated_grads = jax.tree.map(lambda x, y: x + y, accumulated_grads, grads)

                    # Apply accumulated gradients with Adam
                    params, opt_state = apply_accumulated_gradients_adam(
                        params, opt_state, accumulated_grads, optimizer, len(accum_batches)
                    )
                    loss = accumulated_loss / len(accum_batches)
                else:
                    # Use Adam optimizer for single-step updates
                    params, opt_state, loss = compiled_train_step_adam(
                        params,
                        opt_state,
                        accum_batches[0],
                        model,
                        optimizer,
                        trainable_mask,
                    )

            # Reset accumulation
            accum_batches = []
        else:
            # Still accumulating, skip logging but continue loop
            continue

        loss = float(jax.device_get(loss))
        step_time = time.time() - step_start

        metrics_tracker.update({
            "loss": loss,
            "learning_rate": float(lr),
            "step_time": step_time,
        })

        # Log to W&B
        if wandb:
            wandb.log({
                "train/loss": loss,
                "train/learning_rate": lr,
            }, step=step)

        # Log progress
        if step % config.logging.log_every == 0:
            epoch = step // steps_per_epoch + 1
            elapsed = time.time() - start_time
            eta = elapsed / step * (total_steps - step)

            print(
                f"Step {step:5d}/{total_steps} | "
                f"Epoch {epoch:2d}/{config.training.num_epochs} | "
                f"Loss: {loss:.4f} | "
                f"LR: {lr:.6f} | "
                f"ETA: {eta/60:.1f}min"
            )

        # Validation
        if step % config.logging.eval_every == 0:
            # Save validation results for debugging
            val_results_path = os.path.join(
                config.checkpoint_dir,
                f"validation_results_step_{step:06d}.json"
            )

            # Unreplicate params for evaluation if using pmap
            eval_params = unreplicate_params(params) if use_pmap else params

            valid_metrics = evaluate_model(
                model, eval_params, decode_fn, tokenizer,
                valid_dataset, config,
                num_examples=config.eval.num_examples,
                verbose=False,
                save_results_path=val_results_path,
                max_saved_samples=100,
            )

            accuracy = valid_metrics['accuracy']
            print(f"\n  Validation Accuracy: {accuracy:.2%}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_step = step
                print(f"  New best accuracy! Saving best model...")

                # Unreplicate params for saving if using pmap
                save_params = unreplicate_params(params) if use_pmap else params
                save_checkpoint(
                    save_params,
                    step,
                    os.path.join(config.checkpoint_dir, "best"),
                    config,
                    keep_last_n=1,
                )

            if wandb:
                wandb.log({
                    "valid/accuracy": accuracy,
                    "valid/best_accuracy": best_accuracy,
                }, step=step)

            print()

        # Save checkpoint
        if step % config.logging.checkpoint_every == 0:
            # Unreplicate params for saving if using pmap
            save_params = unreplicate_params(params) if use_pmap else params
            save_checkpoint(
                save_params,
                step,
                config.checkpoint_dir,
                config,
                keep_last_n=config.logging.max_checkpoints_to_keep,
            )

        # Epoch boundary
        if step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            epoch_summary = metrics_tracker.compute_epoch_summary()

            print(f"\n{'='*80}")
            print(f"Completed Epoch {epoch}/{config.training.num_epochs}")
            if 'loss_mean' in epoch_summary:
                print(f"  Average Loss: {epoch_summary['loss_mean']:.4f}")
            print(f"  Best Accuracy: {best_accuracy:.2%} (step {best_step})")
            print(f"{'='*80}\n")

            metrics_tracker.reset_epoch()

    # ==========================================================================
    # Final Evaluation
    # ==========================================================================
    print_banner("Step 5: Final Evaluation")

    # Unreplicate params for final evaluation and saving if using pmap
    final_params = unreplicate_params(params) if use_pmap else params

    # Save debug results for final evaluation
    final_results_path = os.path.join(config.checkpoint_dir, "final_eval_results.json")

    final_metrics = evaluate_model(
        model, final_params, decode_fn, tokenizer,
        valid_dataset, config,
        num_examples=None,  # Full validation
        verbose=True,
        save_results_path=final_results_path,
        max_saved_samples=100,
    )

    # ==========================================================================
    # Save Final Model
    # ==========================================================================
    print_banner("Step 6: Saving Final Model")

    final_path = save_checkpoint(
        final_params,
        total_steps,
        config.checkpoint_dir,
        config,
        keep_last_n=config.logging.max_checkpoints_to_keep,
    )

    # ==========================================================================
    # Summary
    # ==========================================================================
    total_time = time.time() - start_time
    print_banner("Production Training Complete!")

    print(f"Experiment: {config.experiment_name}")
    print(f"Total training time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Final validation accuracy: {final_metrics['accuracy']:.2%}")
    print(f"Best validation accuracy: {best_accuracy:.2%} (step {best_step})")
    print(f"\nFinal checkpoint: {final_path}")
    print(f"Best checkpoint: {config.checkpoint_dir}/best/")
    print(f"Config: {config_save_path}")

    # Save final summary
    summary = {
        "experiment_name": config.experiment_name,
        "timestamp": timestamp,
        "total_time_minutes": total_time / 60,
        "total_steps": total_steps,
        "final_accuracy": final_metrics['accuracy'],
        "best_accuracy": best_accuracy,
        "best_step": best_step,
        "final_checkpoint": final_path,
    }
    summary_path = os.path.join(config.logging.output_dir, f"summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    if wandb:
        wandb.finish()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run production training")
    parser.add_argument(
        "--env",
        type=str,
        default=str(PROJECT_ROOT / "envs" / ".env.production"),
        help="Path to .env file",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.env)
    validate_config(config)
    print_config(config)

    # Confirmation
    print("\n" + "=" * 80)
    print("PRODUCTION TRAINING")
    print("=" * 80)
    print("\nThis will start production training with the above configuration.")
    print("Make sure you have:")
    print("  1. Verified training with 01_overfit_test.py")
    print("  2. Analyzed learning curves with 02_train_validate.py")
    print("  3. Tuned hyperparameters based on validation results")

    if not args.yes:
        response = input("\nProceed with production training? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    try:
        train_production(config)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

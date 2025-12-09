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
from src.data import (
    XVRDataset, create_train_iterator, collate_batch, enable_pipeline_debug, _PIPELINE_DEBUG, _log_pipeline,
    # RefCOCOg support
    RefCOCOgDataset, create_refcocog_train_iterator, collate_refcocog_batch,
)
from src.training import (
    create_learning_rate_schedule,
    compiled_train_step_adam,
    compute_loss_and_grads_for_accum,
    accumulate_gradients,  # JIT-compiled gradient accumulation
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
    use_image_grid: bool = False,
):
    """
    Save training batch samples for debugging - EXACTLY like validation does it.

    Simply saves the original text from batch_samples (input_prompt and ground_truth).

    Args:
        use_image_grid: If True, images are in grid format (single combined image)
    """
    import PIL.Image

    # Create output directory
    batch_dir = os.path.join(output_dir, f"train_batch_step_{step:06d}")
    os.makedirs(batch_dir, exist_ok=True)
    images_dir = os.path.join(batch_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # CRITICAL: Use batch_samples directly to avoid image mixing issues
    # batch may be sharded or modified, but batch_samples contains original data
    # Limit to max_samples
    num_samples = min(len(batch_samples), max_samples)

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

        # Save images directly from batch_samples to avoid mixing
        # sample['image'] has shape [num_images, H, W, 3]
        # If use_image_grid=True, num_images=1 and it's a single grid image
        sample_images = sample.get('image', None)
        image_paths = []
        if sample_images is not None and num_images > 0:
            for img_idx in range(num_images):
                img = sample_images[img_idx]

                # Convert to [0, 255]
                if img.min() < 0:
                    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                elif img.max() <= 1.0:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)

                # Use clear naming for grid vs individual images
                if use_image_grid:
                    img_path = os.path.join(images_dir, f"sample_{i:02d}_grid.png")
                    image_paths.append(f"images/sample_{i:02d}_grid.png")
                else:
                    img_path = os.path.join(images_dir, f"sample_{i:02d}_image_{img_idx}.png")
                    image_paths.append(f"images/sample_{i:02d}_image_{img_idx}.png")

                PIL.Image.fromarray(img).save(img_path)

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
    # IMPORTANT: Disable pre-allocation to avoid memory fragmentation
    # JAX by default pre-allocates 90% of GPU memory, which can cause OOM
    # when large contiguous blocks are needed during training.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config.system.xla_mem_fraction)
    
    # CRITICAL: Enable memory pool to allow reuse of memory
    # This allows JAX to reuse memory from accumulated_grads for forward pass
    # Without this, JAX tries to allocate new memory even if old memory can be reused
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Use platform allocator for better memory reuse
    
    # Model debug flags (for big_vision model logging)
    # Set MODEL_DEBUG_EMBEDDING=1 to log embedding details
    # Set MODEL_DEBUG_FORWARD=1 to log forward pass details
    if os.environ.get("MODEL_DEBUG_EMBEDDING", "0") == "1":
        print("  [MODEL DEBUG] Embedding logging enabled")
    if os.environ.get("MODEL_DEBUG_FORWARD", "0") == "1":
        print("  [MODEL DEBUG] Forward pass logging enabled")
    
    # Data pipeline flags
    # Set DISABLE_IMAGES=1 to train without images (text-only training)
    if os.environ.get("DISABLE_IMAGES", "0") == "1":
        print("  [DATA] DISABLE_IMAGES=1: Training without images (text-only mode)")

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
    
    # Helper function to print GPU memory
    def print_gpu_memory(stage):
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode != 0 or not result.stdout.strip():
                print(f'  [GPU Memory] {stage}: Unable to query GPU memory')
                return
            
            # Get first line (first GPU) and split by comma
            first_line = result.stdout.strip().split('\n')[0]
            values = [v.strip() for v in first_line.split(',')]
            
            if len(values) >= 2:
                used, free = values[0], values[1]
                print(f'  [GPU Memory] {stage}: {int(used)//1024}GB used, {int(free)//1024}GB free')
            else:
                print(f'  [GPU Memory] {stage}: Unexpected output format')
        except Exception as e:
            print(f'  [GPU Memory] {stage}: Error querying GPU memory: {e}')
    
    print_gpu_memory("Before model loading")
    model, params, tokenizer, decode_fn = load_paligemma_model(config)
    print_gpu_memory("After model loading")

    trainable_mask = create_trainable_mask(
        params,
        strategy=config.training.trainable_params,
        config=config,
    )
    print_gpu_memory("After trainable mask")
    params = prepare_params_for_training(params, trainable_mask, config)
    print_gpu_memory("After params preparation")

    # ==========================================================================
    # Prepare Data
    # ==========================================================================
    print_banner("Step 2: Preparing Data")

    # Select dataset class based on dataset_type
    dataset_type = config.data.dataset_type.lower()
    print(f"  Dataset type: {dataset_type}")

    if dataset_type == "refcocog":
        # RefCOCOg dataset (single image, bounding box prediction)
        DatasetClass = RefCOCOgDataset
        create_iterator_fn = create_refcocog_train_iterator
        collate_fn = collate_refcocog_batch
        max_images_for_dataset = 1  # RefCOCOg has single images
        print(f"  Using RefCOCOgDataset (single image per sample)")
    else:
        # Default: XVR dataset (multi-image)
        DatasetClass = XVRDataset
        create_iterator_fn = create_train_iterator
        collate_fn = collate_batch
        max_images_for_dataset = config.training.max_images
        print(f"  Using XVRDataset (multi-image: up to {max_images_for_dataset} images)")

    train_dataset = DatasetClass(
        jsonl_path=os.path.join(config.data.base_dir, config.data.train_file),
        image_base_dir=config.data.base_dir,
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        image_size=config.model.img_size,
        max_samples=config.data.max_train_samples,
        shuffle_buffer_size=config.data.shuffle_buffer_size,
    )

    valid_dataset = DatasetClass(
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

    # Create iterator based on dataset type
    if dataset_type == "refcocog":
        train_iterator = create_iterator_fn(
            train_dataset,
            batch_size=config.training.batch_size,
            prompt_prefix=config.data.prompt_prefix,
        )
    else:
        train_iterator = create_iterator_fn(
            train_dataset,
            batch_size=config.training.batch_size,
            prompt_prefix=config.data.prompt_prefix,
            max_images=config.training.max_images,
            use_image_grid=config.data.use_image_grid,
            grid_rows=config.data.grid_rows,
            grid_cols=config.data.grid_cols,
        )

    # ==========================================================================
    # Setup Training
    # ==========================================================================
    print_banner("Step 3: Setting Up Training")

    # Account for gradient accumulation in step calculation
    effective_batch_size = config.training.batch_size * config.training.gradient_accumulation_steps
    effective_steps_per_epoch = train_dataset.num_samples // effective_batch_size
    total_effective_steps = effective_steps_per_epoch * config.training.num_epochs
    
    # Actual steps (number of batches to process) = effective_steps * accum_steps
    # This is what the loop will actually iterate over
    actual_steps_per_epoch = train_dataset.num_samples // config.training.batch_size
    total_steps = actual_steps_per_epoch * config.training.num_epochs

    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Max images per sample: {config.training.max_images}")
    print(f"  Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    
    # Warn if gradient accumulation is not being used
    if config.training.gradient_accumulation_steps == 1:
        print(f"  ⚠️  WARNING: Gradient accumulation is disabled (steps=1)")
        print(f"     Consider setting GRADIENT_ACCUMULATION_STEPS > 1 to increase effective batch size")
    else:
        print(f"  ✅ Gradient accumulation enabled: {config.training.gradient_accumulation_steps} steps")
        print(f"     This allows using smaller batch_size ({config.training.batch_size}) while")
        print(f"     maintaining effective batch size of {effective_batch_size}")
    # Estimate memory usage
    # Each image: 224x224x3 = 150KB (float32) or 75KB (float16)
    # With 6 images per sample and batch_size=8: 8 * 6 * 150KB = 7.2MB just for images
    # But with gradients and activations, memory usage is much higher
    estimated_memory_per_batch_gb = (config.training.batch_size * config.training.max_images * 224 * 224 * 3 * 4) / (1024**3) * 10  # Rough estimate
    print(f"  [WARNING] Estimated memory per batch: ~{estimated_memory_per_batch_gb:.2f}GB (rough estimate)")
    # DEBUG: Verify gradient accumulation is set correctly
    env_val = os.getenv("GRADIENT_ACCUMULATION_STEPS", "NOT SET")
    print(f"  [DEBUG] GRADIENT_ACCUMULATION_STEPS env var: {env_val}")
    print(f"  [DEBUG] config.training.gradient_accumulation_steps: {config.training.gradient_accumulation_steps}")
    print(f"  Effective steps per epoch: {effective_steps_per_epoch} (update steps)")
    print(f"  Actual steps per epoch: {actual_steps_per_epoch} (batch processing steps)")
    print(f"  Total effective steps: {total_effective_steps} (total updates)")
    print(f"  Total actual steps: {total_steps} (total batches to process)")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Optimizer: AdamW")
    print(f"  Estimated time: ~{total_steps * 0.25 / 60:.1f} minutes (based on actual steps)")

    # Create AdamW optimizer with warmup + cosine decay (matches paligemma2 PyTorch)
    # Use effective_steps for optimizer (number of actual updates)
    optimizer = create_optimizer(
        learning_rate=config.training.learning_rate,
        total_steps=total_effective_steps,  # Use effective steps for LR schedule
        warmup_percent=config.training.warmup_percent,
        weight_decay=0.0,  # No weight decay for fine-tuning
        max_grad_norm=config.training.max_grad_norm,
    )

    # Keep lr_schedule for logging purposes only
    lr_schedule = create_learning_rate_schedule(
        base_learning_rate=config.training.learning_rate,
        total_steps=total_effective_steps,  # Use effective steps for LR schedule
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
        print_gpu_memory("Before params device_put")

        # Move params to single GPU (avoid device mismatch)
        first_device = jax.devices()[0]
        params = jax.device_put(params, first_device)
        print_gpu_memory("After params device_put")
        trainable_mask = jax.device_put(trainable_mask, first_device)
        print(f"  Moved params to single GPU: {first_device}")

        # Initialize optimizer state after moving params
        opt_state = create_optimizer_state(optimizer, params)
        print(f"  Optimizer state initialized")
        print_gpu_memory("After optimizer state init")

        data_sharding = create_data_sharding(config)

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    print_banner("Step 4: Training")

    print("Starting production training...")
    print_gpu_memory("Before training loop")
    
    # Enable pipeline debugging for first few samples if DEBUG mode
    if os.environ.get("DATA_PIPELINE_DEBUG", "0") == "1":
        enable_pipeline_debug(max_samples=3)
        print("\n[PIPELINE DEBUG] Enabled - will log first 3 samples in detail\n")

        # Debug: Check embedding generation with a test batch
        print("\n[EMBEDDING DEBUG] Checking embedding generation...")
        test_samples = [next(train_iterator) for _ in range(config.training.batch_size)]
        test_batch = collate_fn(
            test_samples,
            max_images=max_images_for_dataset,
            image_size=config.model.img_size,
        )
        test_batch = shard_batch(test_batch, data_sharding, config)
        
        # Run model without training to check embeddings
        test_imgs = test_batch["image"]
        test_txts = test_batch["text"]
        test_mask_ar = test_batch["mask_ar"]
        test_num_images = test_batch.get("num_images", None)
        
        print(f"  Input shapes:")
        print(f"    images: {test_imgs.shape}")
        print(f"    text: {test_txts.shape}")
        print(f"    mask_ar: {test_mask_ar.shape}")
        print(f"    num_images: {test_num_images}")
        
        # Get model output with auxiliary info
        text_logits, out = model.apply(
            {"params": params},
            test_imgs,
            test_txts[:, :-1],
            test_mask_ar[:, :-1],
            num_images=test_num_images,
            train=False
        )
        
        print(f"  Output shapes:")
        print(f"    text_logits: {text_logits.shape}")
        if "img/zimg" in out:
            zimg = out["img/zimg"]
            print(f"    image_embedding (zimg): {zimg.shape}")
            print(f"      -> {zimg.shape[1]} image tokens per sample")
            print(f"      -> {zimg.shape[1] // 256} images worth of tokens")
        
        # Check if input_mask is properly set for padded images
        if "attn_mask" in out:
            attn_mask = out["attn_mask"]
            print(f"  Attention mask check:")
            print(f"    attn_mask shape: {attn_mask.shape}")
            # Check mask for each sample
            for i in range(min(2, len(test_num_images))):
                num_imgs = test_num_images[i]
                valid_img_tokens = num_imgs * 256
                total_img_tokens = 6 * 256  # max_images * tokens_per_image
                # The attention mask should have False for padded image tokens
                print(f"    Sample {i}: {num_imgs} images")
                print(f"      -> Expected: {valid_img_tokens} valid, {total_img_tokens - valid_img_tokens} masked")
        print(f"  Expected:")
        print(f"    text_logits should be [{test_imgs.shape[0]}, {test_txts.shape[1]-1}, vocab_size]")
        if test_num_images is not None:
            for i in range(min(2, len(test_num_images))):
                print(f"    Sample {i}: {test_num_images[i]} images -> {test_num_images[i] * 256} valid image tokens")
        print("[EMBEDDING DEBUG] Done\n")
        
        # Recreate iterator since we consumed some samples
        if dataset_type == "refcocog":
            train_iterator = create_iterator_fn(
                train_dataset,
                batch_size=config.training.batch_size,
                prompt_prefix=config.data.prompt_prefix,
            )
        else:
            train_iterator = create_iterator_fn(
                train_dataset,
                batch_size=config.training.batch_size,
                prompt_prefix=config.data.prompt_prefix,
                max_images=config.training.max_images,
                use_image_grid=config.data.use_image_grid,
                grid_rows=config.data.grid_rows,
                grid_cols=config.data.grid_cols,
            )
    
    print("")
    start_time = time.time()

    # Gradient accumulation: process batches one at a time and accumulate gradients
    # This avoids storing all batches in memory at once
    accum_steps = config.training.gradient_accumulation_steps
    effective_step = 0
    accum_counter = 0  # Counter for current accumulation
    accumulated_grads = None
    accumulated_loss = 0.0

    for step in range(1, total_steps + 1):
        step_start = time.time()

        # Get batch - collect batch_size samples and use proper collation
        # This handles variable number of images per sample correctly
        batch_samples = [next(train_iterator) for _ in range(config.training.batch_size)]
        batch = collate_fn(
            batch_samples,
            max_images=max_images_for_dataset,
            image_size=config.model.img_size,
        )
        
        # Validate batch images match original samples (if enabled)
        # Set VALIDATE_BATCH_IMAGES=1 in environment to enable
        if os.environ.get("VALIDATE_BATCH_IMAGES", "0") == "1":
            from src.data import _validate_batch_images
            _validate_batch_images(
                batch['image'],
                batch_samples,
                batch['num_images'],
                max_images_for_dataset,
            )

        # DEBUG: Log batch information (occasionally)
        import random
        if step % 100 == 0 and random.random() < 0.1:  # Every 100 steps, 10% chance
            from src.data import get_image_token_count
            print(f"\n[DEBUG training loop] Step {step}:")
            print(f"  batch['image'].shape: {batch['image'].shape}")  # [B, T, H, W, 3]
            print(f"  batch['text'].shape: {batch['text'].shape}")  # [B, seq_len]
            print(f"  batch['num_images']: {batch['num_images']}")  # [B] actual image counts
            image_tokens_per_image = get_image_token_count(config.model.img_size, patch_size=14)
            for i in range(min(3, len(batch['num_images']))):  # Show first 3 samples
                num_imgs = batch['num_images'][i]
                total_img_tokens = image_tokens_per_image * num_imgs
                text_tokens = np.count_nonzero(batch['text'][i])
                total_tokens = total_img_tokens + text_tokens
                print(f"  Sample {i}: {num_imgs} images -> {total_img_tokens} img tokens + {text_tokens} text tokens = {total_tokens} total")

        # Save training batch BEFORE sharding (to avoid image mixing issues)
        # This ensures batch and batch_samples are from the same iteration
        if step % config.logging.log_every == 0:
            try:
                # Create a copy of batch before sharding for saving
                batch_for_save = {
                    'image': batch['image'].copy(),  # Copy to avoid reference issues
                    'num_images': batch['num_images'].copy(),
                }
                save_training_batch(
                    batch=batch_for_save,
                    batch_samples=batch_samples,
                    step=step,
                    loss=0.0,
                    decode_fn=decode_fn,
                    tokenizer=tokenizer,
                    output_dir=config.checkpoint_dir,
                    max_samples=4,
                    use_image_grid=config.data.use_image_grid,
                )
            except Exception as e:
                print(f"  Warning: Failed to save training batch: {e}")

        # Shard batch based on single/multi GPU mode
        if use_pmap:
            batch = shard_batch_for_pmap(batch, num_devices)
        else:
            batch = shard_batch(batch, data_sharding, config)

        # Pipeline logging - after sharding
        if _PIPELINE_DEBUG and step <= 13:
            _log_pipeline("7.SHARD", "=" * 60)
            _log_pipeline("7.SHARD", f"Step {step}: Batch after sharding")
            _log_pipeline("7.SHARD", f"  batch['image'].shape: {batch['image'].shape}")
            _log_pipeline("7.SHARD", f"  batch['text'].shape: {batch['text'].shape}")
            _log_pipeline("7.SHARD", f"  batch['mask_ar'].shape: {batch['mask_ar'].shape}")
            _log_pipeline("7.SHARD", f"  batch['mask_loss'].shape: {batch['mask_loss'].shape}")
            _log_pipeline("7.SHARD", f"  batch['num_images']: {batch['num_images']}")
            
            # Check each sample in batch
            for b_idx in range(min(2, batch['image'].shape[0])):
                num_imgs = batch['num_images'][b_idx]
                text_nonzero = np.count_nonzero(batch['text'][b_idx])
                loss_nonzero = np.count_nonzero(batch['mask_loss'][b_idx])
                _log_pipeline("7.SHARD", f"  Batch item {b_idx}: {num_imgs} images, {text_nonzero} text tokens, {loss_nonzero} loss tokens")
            _log_pipeline("7.SHARD", "=" * 60)

        # Process batch and accumulate gradients (memory efficient)
        accum_counter += 1
        
        # DEBUG: Log accumulation status
        if step % 10 == 0 or accum_counter >= accum_steps:
            should_update = accum_counter >= accum_steps or step == total_steps
            print(f"[DEBUG accum] step={step}, accum_counter={accum_counter}/{accum_steps}, should_update={should_update}")
            if accum_steps > 1 and accum_counter == 1:
                print(f"  ✅ Gradient accumulation active: will accumulate {accum_steps} batches before update")
        
        # Compute loss and gradients for this batch
        if use_pmap:
            # Multi-GPU: can't do gradient accumulation easily, so use single batch
            if accum_steps == 1:
                effective_step += 1
                lr = lr_schedule(effective_step)
                
                # Note: save_training_batch is called before shard_batch (see line 735-750)
                # to avoid image mixing issues with sharded batches
                
                params, opt_state, loss = pmap_train_step(
                    params, opt_state, batch, trainable_mask
                )
                loss = loss[0]
                accum_counter = 0
            else:
                raise NotImplementedError(
                    "Gradient accumulation with pmap is not yet supported. "
                    "Please set GRADIENT_ACCUMULATION_STEPS=1 or USE_PMAP=false"
                )
        else:
            # Single-GPU: accumulate gradients
            if accum_steps > 1:
                # Compute gradients for this batch (JIT-compiled)
                # JAX's memory pool should allow reuse of accumulated_grads memory for forward pass
                # If this still causes OOM, we'll need to use CPU offloading
                loss, grads = compute_loss_and_grads_for_accum(params, batch, model, trainable_mask)

                # ===== DEBUG: Comprehensive training checks =====
                if os.environ.get("DEBUG_TRAINING", "0") == "1":
                    try:
                        from src.debug_utils import run_comprehensive_check

                        # Get current learning rate
                        current_lr = lr_schedule(effective_step + 1) if effective_step + 1 <= total_effective_steps else lr_schedule(total_effective_steps)

                        # Run all checks (only for first few steps or occasionally)
                        if step <= 10 or step % 100 == 1:
                            run_comprehensive_check(
                                batch=batch,
                                params=params,
                                grads=grads,
                                loss=float(loss),
                                step=step,
                                learning_rate=current_lr,
                            )
                    except Exception as e:
                        print(f"[DEBUG] Error in debug checks: {e}")
                        import traceback
                        traceback.print_exc()
                # ===== END DEBUG =====

                # Convert loss to Python float immediately to free JAX array memory
                loss_float = float(jax.device_get(loss))
                accumulated_loss += loss_float
                
                # Accumulate gradients on GPU using JIT-compiled function
                # This is fast because everything stays on GPU
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    # Use JIT-compiled accumulation (GPU-to-GPU, very fast)
                    accumulated_grads = accumulate_gradients(accumulated_grads, grads)
                
                # Explicitly free gradient memory (JAX will handle this, but being explicit helps)
                del grads
                del loss
                
                # Force garbage collection periodically to help JAX free memory
                # Only do this every few steps to avoid overhead
                if accum_counter % 4 == 0:
                    import gc
                    gc.collect()
                
                # Check if we should update
                should_update = accum_counter >= accum_steps or step == total_steps
                
                if should_update:
                    effective_step += 1
                    lr = lr_schedule(effective_step)

                    # Note: save_training_batch is called before shard_batch (see line 735-750)
                    # to avoid image mixing issues with sharded batches

                    # Apply accumulated gradients with Adam
                    # accumulated_grads is already on GPU, so no transfer needed
                    params, opt_state = apply_accumulated_gradients_adam(
                        params, opt_state, accumulated_grads, optimizer, accum_counter
                    )
                    loss = accumulated_loss / accum_counter

                    # Explicitly free accumulated gradients
                    del accumulated_grads
                    accumulated_grads = None
                    accumulated_loss = 0.0
                    accum_counter = 0

                    # Force garbage collection after update
                    import gc
                    gc.collect()
                else:
                    # Still accumulating, skip the rest of the loop
                    # (loss logging happens only after update)
                    continue
            else:
                # Single-step update (no accumulation)
                effective_step += 1
                lr = lr_schedule(effective_step)
                
                # Note: save_training_batch is called before shard_batch (see line 735-750)
                # to avoid image mixing issues with sharded batches
                
                params, opt_state, loss = compiled_train_step_adam(
                    params,
                    opt_state,
                    batch,
                    model,
                    optimizer,
                    trainable_mask,
                )

        loss = float(jax.device_get(loss))
        
        # Pipeline logging - after training step
        if _PIPELINE_DEBUG and step <= 13:
            _log_pipeline("8.TRAIN", "=" * 60)
            _log_pipeline("8.TRAIN", f"Step {step}: Training step complete")
            _log_pipeline("8.TRAIN", f"  Loss: {loss:.6f}")
            _log_pipeline("8.TRAIN", f"  Learning rate: {lr:.2e}")
            _log_pipeline("8.TRAIN", f"  Effective step: {effective_step}")
            
            # Detailed loss breakdown for understanding
            _log_pipeline("8.TRAIN", f"  --- Loss Calculation Breakdown ---")
            _log_pipeline("8.TRAIN", f"  batch['image'].shape: {batch['image'].shape}")
            _log_pipeline("8.TRAIN", f"  batch['text'].shape: {batch['text'].shape}")
            _log_pipeline("8.TRAIN", f"  batch['num_images']: {batch['num_images']}")
            
            # Show mask_loss details per sample
            for b_idx in range(min(2, len(batch['num_images']))):
                num_imgs = batch['num_images'][b_idx]
                mask_loss_nonzero = np.count_nonzero(batch['mask_loss'][b_idx])
                text_nonzero = np.count_nonzero(batch['text'][b_idx])
                
                # Calculate expected token counts
                img_tokens = num_imgs * 256  # 256 tokens per image
                _log_pipeline("8.TRAIN", f"  Sample {b_idx}:")
                _log_pipeline("8.TRAIN", f"    num_images: {num_imgs} -> {img_tokens} image tokens")
                _log_pipeline("8.TRAIN", f"    text tokens (nonzero): {text_nonzero}")
                _log_pipeline("8.TRAIN", f"    mask_loss nonzero (trainable): {mask_loss_nonzero}")
                _log_pipeline("8.TRAIN", f"    total sequence: {img_tokens} + {text_nonzero} = {img_tokens + text_nonzero} tokens")
            
            _log_pipeline("8.TRAIN", f"  --- Expected Behavior ---")
            _log_pipeline("8.TRAIN", f"  - Image tokens: NOT trained (mask_loss=0)")
            _log_pipeline("8.TRAIN", f"  - Prefix text: NOT trained (mask_loss=0)")
            _log_pipeline("8.TRAIN", f"  - Answer tokens: TRAINED (mask_loss=1)")
            _log_pipeline("8.TRAIN", f"  - Loss is computed ONLY on answer tokens ({np.sum(batch['mask_loss'])} total)")
            _log_pipeline("8.TRAIN", "=" * 60)
            _log_pipeline("8.TRAIN", "")
        step_time = time.time() - step_start

        metrics_tracker.update({
            "loss": loss,
            "learning_rate": float(lr),
            "step_time": step_time,
        })

        # Log to W&B (use effective_step for x-axis)
        if wandb:
            wandb.log({
                "train/loss": loss,
                "train/learning_rate": lr,
                "train/step": step,  # Also log raw step for reference
            }, step=effective_step)

        # Log progress (based on effective step)
        if effective_step % config.logging.log_every == 0:
            # Calculate epoch based on effective steps
            epoch = effective_step // effective_steps_per_epoch + 1
            elapsed = time.time() - start_time
            eta = elapsed / effective_step * (total_effective_steps - effective_step) if effective_step > 0 else 0

            # Show both actual step and effective step
            print(
                f"Step {step:5d}/{total_steps} (effective: {effective_step:5d}/{total_effective_steps}) | "
                f"Epoch {epoch:2d}/{config.training.num_epochs} | "
                f"Loss: {loss:.4f} | "
                f"LR: {lr:.2e} | "  # Use scientific notation for very small LR
                f"ETA: {eta/60:.1f}min"
            )

        # Validation (based on effective step)
        if effective_step % config.logging.eval_every == 0 and effective_step > 0:
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
                }, step=effective_step)

            print()

        # Save checkpoint (based on effective step)
        if effective_step % config.logging.checkpoint_every == 0 and effective_step > 0:
            # Unreplicate params for saving if using pmap
            save_params = unreplicate_params(params) if use_pmap else params
            save_checkpoint(
                save_params,
                step,
                config.checkpoint_dir,
                config,
                keep_last_n=config.logging.max_checkpoints_to_keep,
            )

        # Epoch boundary (based on actual steps)
        if step % actual_steps_per_epoch == 0:
            epoch = step // actual_steps_per_epoch
            epoch_summary = metrics_tracker.compute_epoch_summary()

            print(f"\n{'='*80}")
            print(f"Completed Epoch {epoch}/{config.training.num_epochs}")
            print(f"  Actual steps processed: {step}")
            print(f"  Effective updates: {effective_step}")
            if 'loss_mean' in epoch_summary:
                print(f"  Average Loss: {epoch_summary['loss_mean']:.4f}")
            print(f"  Best Accuracy: {best_accuracy:.2%} (effective step {best_step})")
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
        total_effective_steps,  # Use effective step for checkpoint naming
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
        "total_actual_steps": total_steps,  # Number of batches processed
        "total_effective_steps": total_effective_steps,  # Number of parameter updates
        "final_accuracy": final_metrics['accuracy'],
        "best_accuracy": best_accuracy,
        "best_step": best_step,  # Effective step
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

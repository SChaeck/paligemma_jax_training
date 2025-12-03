"""
Model utilities for PaliGemma fine-tuning.

This module provides functions for loading models, creating trainable masks,
and saving/loading checkpoints.
"""

import functools
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import sentencepiece

from .config import Config


def setup_big_vision(config: Config):
    """Add big_vision to path if needed."""
    if config.system.big_vision_path not in sys.path:
        if os.path.exists(config.system.big_vision_path):
            sys.path.insert(0, config.system.big_vision_path)


def download_checkpoint_if_needed(checkpoint_path: str, checkpoint_url: Optional[str] = None) -> str:
    """
    Download checkpoint from URL if it doesn't exist locally.

    Args:
        checkpoint_path: Local path where checkpoint should be stored
        checkpoint_url: Optional URL to download from if local file doesn't exist

    Returns:
        Path to the checkpoint (either existing or downloaded)
    """
    checkpoint_path = Path(checkpoint_path)

    # If file exists, use it
    if checkpoint_path.exists():
        print(f"  Using existing checkpoint: {checkpoint_path}")
        return str(checkpoint_path)

    # If no URL provided, check for symlink target or raise error
    if checkpoint_url is None:
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            f"Please download the checkpoint or provide MODEL_CHECKPOINT_URL in .env"
        )

    # Download from URL
    print(f"  Checkpoint not found locally, downloading from URL...")
    print(f"  URL: {checkpoint_url}")
    print(f"  Target: {checkpoint_path}")

    import requests
    from tqdm import tqdm

    # Create parent directory if needed
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Download with progress bar
    response = requests.get(checkpoint_url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(checkpoint_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

    print(f"  Downloaded successfully ({checkpoint_path.stat().st_size / 1e9:.2f} GB)")
    return str(checkpoint_path)


def load_paligemma_model(
    config: Config,
) -> Tuple[Any, Dict, Any, Any]:
    """
    Load PaliGemma model, parameters, tokenizer, and decode function.

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, params, tokenizer, decode_fn)
    """
    # Ensure big_vision is in path
    setup_big_vision(config)

    # Import big_vision modules
    from big_vision.models.proj.paligemma import paligemma
    from big_vision.trainers.proj.paligemma import predict_fns

    print("Loading PaliGemma model...")

    # Build model config
    llm_config = {
        "vocab_size": config.model.vocab_size,
        "variant": config.model.llm_variant,
    }
    if config.model.llm_variant.startswith("gemma2"):
        llm_config["final_logits_softcap"] = 30.0

    model_config = ml_collections.FrozenConfigDict({
        "llm": llm_config,
        "img": {
            "variant": config.model.img_variant,
            "pool_type": config.model.img_pool_type,
            "scan": config.model.img_scan,
            "dtype_mm": config.model.img_dtype_mm,
        }
    })

    # Initialize model
    model = paligemma.Model(**model_config)
    print(f"  Model initialized")

    # Load tokenizer
    tokenizer = sentencepiece.SentencePieceProcessor(config.model.tokenizer_path)
    print(f"  Tokenizer loaded")

    # Get checkpoint path (download if needed)
    checkpoint_url = getattr(config.model, 'checkpoint_url', None)
    checkpoint_path = download_checkpoint_if_needed(
        config.model.checkpoint_path,
        checkpoint_url
    )

    # Load parameters
    print(f"  Loading parameters (this may take 1-2 minutes)...")
    params = paligemma.load(None, checkpoint_path, model_config)
    print(f"  Parameters loaded")

    # Create decode function
    decode_fn = predict_fns.get_all(model)['decode']
    decode = functools.partial(
        decode_fn,
        devices=jax.devices(),
        eos_token=tokenizer.eos_id()
    )
    print(f"  Decode function created")

    return model, params, tokenizer, decode


def create_trainable_mask(
    params: Dict,
    strategy: str = "attention_only",
    config: Optional[Config] = None,
) -> Dict:
    """
    Create a mask indicating which parameters should be trained.

    Args:
        params: Model parameters
        strategy: Training strategy
            - "attention_only": Only train attention layers (memory efficient)
            - "full_llm": Train all language model parameters
            - "full_model": Train entire model including vision encoder
        config: Optional config for big_vision path

    Returns:
        Boolean mask with same structure as params
    """
    if config:
        setup_big_vision(config)

    import big_vision.utils

    def is_trainable(name: str, param: Any) -> bool:
        """Determine if a parameter should be trainable."""
        if strategy == "attention_only":
            if name.startswith("llm/layers/attn/"):
                return True
            return False

        elif strategy == "full_llm":
            if name.startswith("llm/"):
                return True
            return False

        elif strategy == "full_model":
            return True

        else:
            raise ValueError(f"Unknown training strategy: {strategy}")

    # Create mask using big_vision utility
    trainable_mask = big_vision.utils.tree_map_with_names(
        is_trainable,
        params
    )

    # Print summary
    total_params = count_parameters(params)
    trainable_params = count_parameters(params, trainable_mask)
    frozen_params = total_params - trainable_params

    print(f"\nParameter Summary:")
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  Frozen parameters:     {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")

    return trainable_mask


def prepare_params_for_training(
    params: Dict,
    trainable_mask: Dict,
    config: Optional[Config] = None,
) -> Dict:
    """
    Prepare parameters for training by sharding and casting.

    Args:
        params: Model parameters
        trainable_mask: Boolean mask of trainable parameters
        config: Optional config for big_vision path

    Returns:
        Prepared parameters
    """
    if config:
        setup_big_vision(config)

    import big_vision.utils
    import big_vision.sharding

    # Create mesh
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))

    # Infer sharding strategy
    params_sharding = big_vision.sharding.infer_sharding(
        params,
        strategy=[('.*', 'fsdp(axis="data")')],
        mesh=mesh
    )

    # Suppress warnings about donated buffers
    warnings.filterwarnings(
        "ignore",
        message="Some donated buffers were not usable"
    )

    # Cast parameters based on trainability
    @functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
    def maybe_cast_to_f32(params, trainable):
        return jax.tree.map(
            lambda p, m: p.astype(jnp.float32) if m else p.astype(jnp.float16),
            params,
            trainable
        )

    # Process parameters sequentially to avoid OOM
    params_flat, treedef = jax.tree.flatten(params)
    sharding_leaves = jax.tree.leaves(params_sharding)
    trainable_leaves = jax.tree.leaves(trainable_mask)

    for idx, (sharding, trainable) in enumerate(zip(sharding_leaves, trainable_leaves)):
        params_flat[idx] = big_vision.utils.reshard(params_flat[idx], sharding)
        params_flat[idx] = maybe_cast_to_f32(params_flat[idx], trainable)
        params_flat[idx].block_until_ready()

    params = jax.tree.unflatten(treedef, params_flat)

    return params


def count_parameters(params: Dict, mask: Optional[Dict] = None) -> int:
    """
    Count total number of parameters.

    Args:
        params: Model parameters
        mask: Optional boolean mask (if provided, only count True params)

    Returns:
        Total number of parameters
    """
    def count_fn(param, m=True):
        if mask is not None and not m:
            return 0
        return param.size

    if mask is not None:
        counts = jax.tree.map(count_fn, params, mask)
    else:
        counts = jax.tree.map(count_fn, params)

    return sum(jax.tree.leaves(counts))


def parameter_overview(params: Dict, config: Optional[Config] = None) -> None:
    """Print an overview of model parameters."""
    if config:
        setup_big_vision(config)

    import big_vision.utils

    print("\nModel Parameters:")
    print("=" * 120)
    print(f"{'Parameter Name':<80} {'Shape':<25} {'Dtype':<15}")
    print("=" * 120)

    for path, arr in big_vision.utils.tree_flatten_with_names(params)[0]:
        print(f"{path:<80} {str(arr.shape):<25} {str(arr.dtype):<15}")

    print("=" * 120)


def save_checkpoint(
    params: Dict,
    step: int,
    checkpoint_dir: str,
    config: Optional[Config] = None,
    keep_last_n: int = 3,
) -> str:
    """
    Save model checkpoint.

    Args:
        params: Model parameters to save
        step: Training step number
        checkpoint_dir: Directory to save checkpoints
        config: Optional configuration to save alongside
        keep_last_n: Keep only the last N checkpoints

    Returns:
        Path to saved checkpoint
    """
    if config:
        setup_big_vision(config)

    import big_vision.utils

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_{step:06d}.npz"

    print(f"Saving checkpoint to {checkpoint_path}...")

    # Convert to numpy and save
    params_numpy = jax.device_get(params)

    # Flatten parameters for saving
    flat_params = {}
    for path, arr in big_vision.utils.tree_flatten_with_names(params_numpy)[0]:
        flat_params[path] = arr

    # Save parameters
    np.savez(checkpoint_path, **flat_params)

    # Save config if provided
    if config is not None:
        import json
        config_path = checkpoint_dir / f"config_{step:06d}.json"
        config_dict = {
            "experiment_name": config.experiment_name,
            "model": config.model.__dict__,
            "training": config.training.__dict__,
            "data": config.data.__dict__,
            "logging": config.logging.__dict__,
            "eval": config.eval.__dict__,
            "system": config.system.__dict__,
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    print(f"  Checkpoint saved")

    # Clean up old checkpoints
    _cleanup_old_checkpoints(checkpoint_dir, keep_last_n)

    return str(checkpoint_path)


def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load model checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    loaded = np.load(checkpoint_path)
    params = dict(loaded)
    print(f"  Checkpoint loaded")
    return params


def _cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int):
    """Remove old checkpoints, keeping only the last N."""
    if keep_last_n <= 0:
        return

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.npz"))

    if len(checkpoints) > keep_last_n:
        for ckpt in checkpoints[:-keep_last_n]:
            print(f"  Removing old checkpoint: {ckpt.name}")
            ckpt.unlink()

            config_file = checkpoint_dir / ckpt.name.replace("checkpoint_", "config_").replace(".npz", ".json")
            if config_file.exists():
                config_file.unlink()

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


def download_checkpoint_if_needed(
    checkpoint_path: str, 
    checkpoint_url: Optional[str] = None,
    kaggle_handle: Optional[str] = None
) -> str:
    """
    Download checkpoint from URL or Kaggle if it doesn't exist locally.

    Args:
        checkpoint_path: Local path where checkpoint should be stored, or Kaggle handle
        checkpoint_url: Optional URL to download from if local file doesn't exist
        kaggle_handle: Optional Kaggle handle (e.g., "google/paligemma/jax/paligemma-3b-pt-224")

    Returns:
        Path to the checkpoint (either existing or downloaded)
    """
    # Check if checkpoint_path is a Kaggle handle (contains '/' but not a file path)
    is_kaggle_handle = (
        '/' in checkpoint_path and 
        not checkpoint_path.startswith('.') and 
        not checkpoint_path.startswith('/') and
        not Path(checkpoint_path).exists() and
        not checkpoint_path.endswith('.npz')
    )
    
    # If it looks like a Kaggle handle, use kagglehub to download
    if is_kaggle_handle or kaggle_handle:
        handle = kaggle_handle or checkpoint_path
        print(f"  Detected Kaggle handle: {handle}")
        print(f"  Downloading checkpoint from Kaggle (this may take a few minutes)...")
        
        try:
            import kagglehub
            # Download from Kaggle
            kaggle_path = kagglehub.model_download(handle)
            kaggle_path = Path(kaggle_path)
            
            # Find the .npz file in the downloaded directory
            npz_files = list(kaggle_path.glob("*.npz"))
            if npz_files:
                # Use the first .npz file found
                downloaded_path = npz_files[0]
                print(f"  Downloaded checkpoint: {downloaded_path}")
                return str(downloaded_path)
            else:
                # If no .npz found, check if the handle itself is a file path
                if kaggle_path.is_file():
                    print(f"  Using checkpoint: {kaggle_path}")
                    return str(kaggle_path)
                else:
                    raise FileNotFoundError(
                        f"No .npz file found in Kaggle download: {kaggle_path}. "
                        f"Please specify the full path to the checkpoint file."
                    )
        except ImportError:
            raise ImportError(
                "kagglehub is required to download from Kaggle. "
                "Install it with: pip install kagglehub"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download checkpoint from Kaggle: {e}. "
                f"Please download manually or provide MODEL_CHECKPOINT_URL."
            )
    
    # Otherwise, treat as local file path
    checkpoint_path = Path(checkpoint_path)

    # If file exists, use it
    if checkpoint_path.exists():
        print(f"  Using existing checkpoint: {checkpoint_path}")
        return str(checkpoint_path)

    # If no URL provided, check for symlink target or raise error
    if checkpoint_url is None:
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            f"Please download the checkpoint, provide MODEL_CHECKPOINT_URL, "
            f"or use a Kaggle handle (e.g., 'google/paligemma/jax/paligemma-3b-pt-224')"
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
    kaggle_handle = getattr(config.model, 'kaggle_handle', None)
    checkpoint_path = download_checkpoint_if_needed(
        config.model.checkpoint_path,
        checkpoint_url,
        kaggle_handle
    )

    # Load parameters
    print(f"  Loading parameters (this may take 1-2 minutes)...")
    params = paligemma.load(None, checkpoint_path, model_config)
    print(f"  Parameters loaded")

    # Create decode function
    # Use only the devices that training uses (respects USE_PMAP setting)
    # If USE_PMAP=false, use only first device to match single-GPU training
    decode_fn = predict_fns.get_all(model)['decode']
    if config.training.use_pmap:
        decode_devices = jax.devices()
    else:
        # Single device training - use only first device for eval
        decode_devices = jax.devices()[:1]

    decode = functools.partial(
        decode_fn,
        devices=decode_devices,
        eos_token=tokenizer.eos_id()
    )
    print(f"  Decode function created (devices: {decode_devices})")

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

    # Flatten parameters for saving with 'params/' prefix (matches original checkpoint format)
    flat_params = {}
    for path, arr in big_vision.utils.tree_flatten_with_names(params_numpy)[0]:
        # Add 'params/' prefix to match original checkpoint format
        key = f"params/{path}"
        flat_params[key] = arr

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


def get_attention_weights(model, params, batch, layer_indices=None):
    """
    Get actual attention weights from specific layers using the modified model.

    This captures the REAL softmax attention probabilities from the transformer,
    showing how much each token actually attends to other tokens.

    Args:
        model: PaliGemma model
        params: Model parameters
        batch: Prepared inference batch
        layer_indices: List of layer indices to extract attention from.
                      If None, uses last layer only. Use [0, 17] for first and last.

    Returns:
        Dictionary containing:
        - attention_weights: dict mapping layer_idx -> np.array [B, T, S]
        - num_image_tokens: number of image tokens
        - output_dict: full model output dictionary
    """
    image = jnp.array(batch['image'])
    text = jnp.array(batch['text'])
    mask_ar = jnp.array(batch.get('mask_ar'))
    num_images = batch.get('num_images')
    if num_images is not None:
        num_images = jnp.array(num_images)

    # Use the new forward_with_attention method
    text_logits, out, attention_weights = model.apply(
        {'params': params},
        image,
        text,
        mask_ar,
        num_images=num_images,
        train=False,
        layer_indices=layer_indices,
        method=model.forward_with_attention,
    )

    # Convert attention weights to numpy
    attn_weights_np = {}
    for layer_idx, weights in attention_weights.items():
        attn_weights_np[layer_idx] = np.array(jax.device_get(weights))

    return {
        'attention_weights': attn_weights_np,
        'num_image_tokens': int(out.get('num_image_tokens', 0)),
        'text_logits': np.array(jax.device_get(text_logits)),
        'output_dict': {k: np.array(jax.device_get(v)) if hasattr(v, 'shape') else v
                       for k, v in out.items() if k != 'llm/encoded'},
    }


def visualize_attention_weights(model, params, batch, output_path=None, title="Attention Weights", layer_idx=-1):
    """
    Visualize REAL attention weights from the model.

    This uses the modified model to extract actual attention probabilities
    from the specified transformer layer.

    Args:
        model: PaliGemma model
        params: Model parameters
        batch: Prepared inference batch
        output_path: Path to save visualization
        title: Plot title
        layer_idx: Which layer to visualize (-1 for last layer, 0 for first)

    Returns:
        Dictionary with attention info
    """
    import matplotlib.pyplot as plt

    num_images = batch.get('num_images')

    # Convert layer_idx to list (for forward_with_attention)
    # Gemma 2B has 18 layers (0-17)
    if layer_idx == -1:
        layer_indices = [17]  # Last layer
    else:
        layer_indices = [layer_idx]

    # Get REAL attention weights from the model
    result = get_attention_weights(model, params, batch, layer_indices=layer_indices)

    attn_weights_dict = result['attention_weights']
    img_tokens = result['num_image_tokens']

    # Get the attention weights for the requested layer
    actual_layer_idx = layer_indices[0]
    attn_weights = attn_weights_dict[actual_layer_idx][0]  # First sample, shape [T, S]

    # Calculate image token count
    n_img = int(num_images[0]) if num_images is not None else 1

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Full attention map
    ax1 = axes[0, 0]
    # Clip for visualization since attention can have extreme values
    attn_clipped = np.clip(attn_weights, 0, 0.1)
    im1 = ax1.imshow(attn_clipped, cmap='hot', aspect='auto')
    ax1.set_title(f"{title} (Layer {actual_layer_idx})\nReal Attention Weights (clipped at 0.1)")
    ax1.axvline(x=img_tokens-0.5, color='lime', linestyle='--', linewidth=1, label='Image|Text')
    ax1.axhline(y=img_tokens-0.5, color='lime', linestyle='--', linewidth=1)
    ax1.set_xlabel("Key Position (attending TO)")
    ax1.set_ylabel("Query Position (attending FROM)")
    plt.colorbar(im1, ax=ax1, label='Attention Weight')

    # 2. Attention to images vs text (per query position)
    ax2 = axes[0, 1]
    attn_to_img = attn_weights[:, :img_tokens].sum(axis=1)  # [T]
    attn_to_txt = attn_weights[:, img_tokens:].sum(axis=1)  # [T]

    x_pos = np.arange(len(attn_to_img))
    ax2.plot(x_pos, attn_to_img, label='To Image', alpha=0.8)
    ax2.plot(x_pos, attn_to_txt, label='To Text', alpha=0.8)
    ax2.axvline(x=img_tokens, color='red', linestyle='--', label='Image|Text boundary')
    ax2.set_xlabel("Query Position")
    ax2.set_ylabel("Total Attention")
    ax2.set_title("Attention Distribution\n(Total attention to image vs text tokens)")
    ax2.legend()
    ax2.set_xlim(0, len(attn_to_img))

    # 3. First generated token's attention pattern
    ax3 = axes[1, 0]
    # The first token AFTER the question is at position (img_tokens + question_length)
    # For simplicity, we look at position img_tokens + 10 (assuming question is ~10 tokens)
    # But actually, let's find the first position with high attention variance (likely generation position)
    first_gen_pos = img_tokens + 10  # Approximate

    if first_gen_pos < attn_weights.shape[0]:
        first_gen_attn = attn_weights[first_gen_pos, :]  # [S]

        # Plot attention from first generated token to all positions
        ax3.bar(range(len(first_gen_attn)), first_gen_attn, width=1.0, alpha=0.7)
        ax3.axvline(x=img_tokens, color='red', linestyle='--', linewidth=2, label='Image|Text')
        ax3.set_xlabel("Key Position")
        ax3.set_ylabel("Attention Weight")
        ax3.set_title(f"First Generation Token (pos {first_gen_pos})\nAttention to all previous tokens")
        ax3.legend()

        # Add annotation for total attention to images
        total_to_img = first_gen_attn[:img_tokens].sum()
        ax3.text(0.02, 0.98, f"Total to images: {total_to_img:.3f}",
                transform=ax3.transAxes, fontsize=10, verticalalignment='top')
    else:
        ax3.text(0.5, 0.5, "Position out of range", ha='center', va='center', transform=ax3.transAxes)

    # 4. Text→Image attention detail (per image)
    ax4 = axes[1, 1]
    text_start = img_tokens
    text_end = min(img_tokens + 50, attn_weights.shape[0])

    if text_end > text_start and img_tokens > 0 and n_img > 0:
        text_to_img_attn = attn_weights[text_start:text_end, :img_tokens]

        # Sum attention to each image (256 tokens per image)
        tokens_per_image = img_tokens // n_img if n_img > 0 else img_tokens
        attn_per_image = []
        for i in range(n_img):
            start_tok = i * tokens_per_image
            end_tok = min((i + 1) * tokens_per_image, img_tokens)
            attn_per_image.append(text_to_img_attn[:, start_tok:end_tok].sum(axis=1))

        attn_per_image = np.array(attn_per_image).T  # [n_text, n_images]

        im4 = ax4.imshow(attn_per_image, cmap='YlOrRd', aspect='auto')
        ax4.set_title(f"Text→Image REAL Attention\n(Layer {actual_layer_idx}, {n_img} images)")
        ax4.set_xlabel("Image Index")
        ax4.set_ylabel("Text Token (relative from question start)")
        ax4.set_xticks(range(n_img))
        ax4.set_xticklabels([f"Img {i+1}" for i in range(n_img)])
        plt.colorbar(im4, ax=ax4, label='Total Attention')

        # Compute stats
        avg_attn_per_image = attn_per_image.mean(axis=0)
        print(f"      REAL Avg attention per image: {', '.join([f'Img{i+1}={v:.4f}' for i, v in enumerate(avg_attn_per_image)])}")
    else:
        ax4.text(0.5, 0.5, "No text tokens or no images", ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"      Attention visualization saved to: {output_path}")

    plt.close()

    # Compute statistics
    # How much do text tokens attend to image vs other text tokens?
    if text_end > text_start and img_tokens > 0:
        text_to_img_total = attn_weights[text_start:text_end, :img_tokens].sum(axis=1).mean()
        text_to_text_total = attn_weights[text_start:text_end, img_tokens:text_end].sum(axis=1).mean()
    else:
        text_to_img_total = 0.0
        text_to_text_total = 0.0

    return {
        'num_image_tokens': img_tokens,
        'num_images': n_img,
        'layer_idx': actual_layer_idx,
        'text_to_image_attention': float(text_to_img_total),
        'text_to_text_attention': float(text_to_text_total),
        'attention_weights_shape': attn_weights.shape,
    }


def visualize_attention_mask(model, params, batch, output_path=None, title="Attention Mask"):
    """
    Visualize the attention mask to see which tokens can attend to which.

    This shows the *allowed* attention pattern, not the actual attention weights.
    - Image tokens are at the beginning (num_images * 256 tokens)
    - Text tokens follow

    Args:
        model: PaliGemma model
        params: Model parameters
        batch: Prepared inference batch
        output_path: Path to save the visualization (optional)
        title: Title for the plot

    Returns:
        Dictionary with attention mask info
    """
    import matplotlib.pyplot as plt

    image = jnp.array(batch['image'])
    text = jnp.array(batch['text'])
    mask_ar = jnp.array(batch.get('mask_ar'))
    num_images = batch.get('num_images')
    if num_images is not None:
        num_images = jnp.array(num_images)

    # Get embeddings and attention mask via embed_image_and_text
    # Note: mask_ar is a keyword argument, not positional
    (x, input_mask_out, mask_ar_out), out = model.apply(
        {'params': params},
        image,
        text,
        mask_ar=mask_ar,
        num_images=num_images,
        train=False,
        method=model.embed_image_and_text,
    )

    attn_mask = out.get('attn_mask')
    if attn_mask is None:
        # Compute it manually
        from big_vision.models.proj.paligemma.paligemma import make_attn_mask
        attn_mask = make_attn_mask(input_mask_out, mask_ar_out)

    # Get first sample
    attn_mask_np = np.array(jax.device_get(attn_mask[0]))  # [seq_len, seq_len]

    # Calculate image token count
    n_img = int(num_images[0]) if num_images is not None else 1
    img_tokens = n_img * 256  # 256 tokens per image (224/14)^2

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Full attention mask
    ax1 = axes[0]
    im1 = ax1.imshow(attn_mask_np.astype(float), cmap='Blues', aspect='auto')
    ax1.set_title(f"{title}\n(Full Sequence)")
    ax1.set_xlabel("Key Position (attending TO)")
    ax1.set_ylabel("Query Position (attending FROM)")

    # Add region labels
    ax1.axvline(x=img_tokens-0.5, color='red', linestyle='--', linewidth=1, label='Image|Text boundary')
    ax1.axhline(y=img_tokens-0.5, color='red', linestyle='--', linewidth=1)
    plt.colorbar(im1, ax=ax1, label='Can Attend (1=Yes)')

    # Zoomed view: text tokens attending to image tokens
    ax2 = axes[1]
    # Get the region where text attends to images
    text_start = img_tokens
    text_end = min(img_tokens + 50, attn_mask_np.shape[0])  # First 50 text tokens
    img_end = img_tokens

    if text_end > text_start and img_end > 0:
        text_to_img = attn_mask_np[text_start:text_end, :img_end]
        im2 = ax2.imshow(text_to_img.astype(float), cmap='Blues', aspect='auto')
        ax2.set_title(f"Text→Image Attention Pattern\n(First {text_end-text_start} text tokens → {img_end} image tokens)")
        ax2.set_xlabel("Image Token Position")
        ax2.set_ylabel("Text Token Position (relative)")
        plt.colorbar(im2, ax=ax2, label='Can Attend (1=Yes)')
    else:
        ax2.text(0.5, 0.5, "No text tokens", ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"      Attention mask saved to: {output_path}")

    plt.close()

    # Compute statistics
    # How much do text tokens attend to image tokens on average?
    if text_end > text_start and img_end > 0:
        text_to_img_ratio = text_to_img.mean()
    else:
        text_to_img_ratio = 0.0

    return {
        'attn_mask_shape': attn_mask_np.shape,
        'num_image_tokens': img_tokens,
        'num_text_tokens': attn_mask_np.shape[0] - img_tokens,
        'text_to_image_attention_ratio': float(text_to_img_ratio),
    }


def get_first_token_logits(model, params, batch):
    """
    Get logits for the first generated token position.

    This is useful for comparing how different image inputs affect
    the model's output distribution, independent of greedy decoding.

    Args:
        model: PaliGemma model
        params: Model parameters
        batch: Prepared inference batch with 'image', 'text', 'mask_ar', 'num_images'

    Returns:
        Logits array of shape [vocab_size] for first token position
    """
    # Run forward pass to get logits
    # PaliGemma model.__call__ signature: (image, text, mask_ar, num_images=None, train=False)
    image = batch['image']  # [B, T, H, W, 3]
    text = batch['text']    # [B, seq_len]
    mask_ar = batch.get('mask_ar')  # [B, seq_len]
    mask_input = batch.get('mask_input')  # [B, seq_len] - used to find generation start position
    num_images = batch.get('num_images')  # [B]

    # Convert to JAX arrays if needed
    image = jnp.array(image)
    text = jnp.array(text)
    if mask_ar is not None:
        mask_ar = jnp.array(mask_ar)
    if mask_input is not None:
        mask_input = jnp.array(mask_input)
    if num_images is not None:
        num_images = jnp.array(num_images)

    # Forward pass - note: model.__call__ does NOT take mask_input, only mask_ar
    logits, _ = model.apply(
        {'params': params},
        image,
        text,
        mask_ar,  # positional argument
        num_images=num_images,
        train=False,
    )

    # logits shape: [B, seq_len, vocab_size]
    # Find the position after the last input token (where generation starts)
    # This is where mask_input transitions from 1 to 0
    if mask_input is not None:
        # Find last position where mask_input is 1
        input_lengths = jnp.sum(mask_input, axis=-1)  # [B]
        first_gen_pos = input_lengths.astype(jnp.int32) - 1  # -1 because we want logits at last input position
    else:
        # Fallback: use last position
        first_gen_pos = jnp.array([text.shape[1] - 1])

    # Get logits at first generation position for batch item 0
    first_logits = logits[0, first_gen_pos[0], :]  # [vocab_size]

    # Convert to numpy for comparison
    return np.array(jax.device_get(first_logits))

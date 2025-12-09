"""
Comprehensive debugging utilities for training issues.

Enable with DEBUG_TRAINING=1 environment variable.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional


_DEBUG_ENABLED = os.environ.get("DEBUG_TRAINING", "0") == "1"
_CHECK_COUNTER = 0


def is_debug_enabled() -> bool:
    """Check if debugging is enabled."""
    return _DEBUG_ENABLED


def debug_print(message: str, force: bool = False):
    """Print debug message if debugging is enabled."""
    if _DEBUG_ENABLED or force:
        print(f"[DEBUG] {message}")


def print_section(title: str):
    """Print a section header."""
    if _DEBUG_ENABLED:
        print(f"\n{'='*80}")
        print(f"DEBUG CHECK: {title}")
        print('='*80)


def check_learning_rate(learning_rate: float, step: int = 0):
    """
    Check 1: Learning Rate Analysis

    Verify if learning rate is appropriate for fine-tuning.
    """
    if not _DEBUG_ENABLED:
        return

    print_section("Learning Rate Analysis")

    debug_print(f"Current learning rate: {learning_rate:.2e}")
    debug_print(f"Current step: {step}")

    # Check if LR is too high
    if learning_rate > 1e-5:
        debug_print("⚠️  WARNING: Learning rate is HIGH for fine-tuning!")
        debug_print(f"   Typical fine-tuning LR: 1e-6 ~ 1e-5")
        debug_print(f"   Your LR: {learning_rate:.2e}")
        debug_print(f"   This might cause loss to oscillate and not converge!")
    elif learning_rate < 1e-7:
        debug_print("⚠️  WARNING: Learning rate might be TOO LOW!")
        debug_print(f"   Training might be extremely slow")
    else:
        debug_print("✓ Learning rate is in reasonable range for fine-tuning")


def check_batch_data(batch: Dict[str, np.ndarray], step: int = 0):
    """
    Check 2-5: Batch Data Analysis

    - Check image shapes and values
    - Check num_images
    - Check text tokens
    - Check masks
    """
    if not _DEBUG_ENABLED:
        return

    global _CHECK_COUNTER
    _CHECK_COUNTER += 1

    # Only check every N steps to avoid spam
    if _CHECK_COUNTER % 100 != 1 and step > 10:
        return

    print_section(f"Batch Data Analysis (Step {step})")

    # Check 2: Image data
    debug_print("--- Image Data ---")
    images = batch['image']
    debug_print(f"Image shape: {images.shape}")
    debug_print(f"Image dtype: {images.dtype}")

    # Check if images are all zeros
    num_nonzero = np.count_nonzero(images)
    total_elements = images.size
    nonzero_pct = (num_nonzero / total_elements * 100) if total_elements > 0 else 0
    debug_print(f"Non-zero pixels: {num_nonzero:,} / {total_elements:,} ({nonzero_pct:.1f}%)")

    if nonzero_pct < 1.0:
        debug_print("⚠️  WARNING: Images are mostly zeros!")
        debug_print("   This might indicate empty/black images")

    # Image statistics (per sample)
    for i in range(min(2, images.shape[0])):
        img_sample = images[i]
        if len(img_sample.shape) == 4:  # [T, H, W, 3]
            # Multi-image case
            num_imgs = img_sample.shape[0]
            debug_print(f"  Sample {i}: {num_imgs} images")
            for j in range(min(3, num_imgs)):
                img = img_sample[j]
                debug_print(f"    Image {j}: mean={np.mean(img):.3f}, std={np.std(img):.3f}, "
                          f"min={np.min(img):.3f}, max={np.max(img):.3f}")
        else:
            # Single image case
            debug_print(f"  Sample {i}: mean={np.mean(img_sample):.3f}, std={np.std(img_sample):.3f}, "
                      f"min={np.min(img_sample):.3f}, max={np.max(img_sample):.3f}")

    # Check 3: num_images
    debug_print("\n--- num_images Check ---")
    if 'num_images' in batch:
        num_images = batch['num_images']
        debug_print(f"num_images shape: {num_images.shape}")
        debug_print(f"num_images values: {num_images}")

        if np.any(num_images == 0):
            debug_print("⚠️  WARNING: Some samples have num_images=0!")
            debug_print("   This means all image tokens will be masked as invalid")
        else:
            debug_print("✓ All samples have valid images")
    else:
        debug_print("⚠️  WARNING: num_images not in batch!")
        debug_print("   Image token masking might not work correctly")

    # Check 4: Text data
    debug_print("\n--- Text Data ---")
    text = batch['text']
    debug_print(f"Text shape: {text.shape}")
    debug_print(f"Text dtype: {text.dtype}")

    # Count non-padding tokens
    nonzero_per_sample = np.sum(text != 0, axis=-1)
    debug_print(f"Non-padding tokens per sample: {nonzero_per_sample}")

    # Check 5: Masks
    debug_print("\n--- Mask Analysis ---")
    mask_ar = batch['mask_ar']
    mask_loss = batch['mask_loss']

    debug_print(f"mask_ar shape: {mask_ar.shape}")
    debug_print(f"mask_loss shape: {mask_loss.shape}")

    # Count causal and trainable tokens
    for i in range(min(2, mask_ar.shape[0])):
        causal_tokens = np.sum(mask_ar[i])
        trainable_tokens = np.sum(mask_loss[i])
        debug_print(f"  Sample {i}:")
        debug_print(f"    Causal tokens (mask_ar=1): {causal_tokens}")
        debug_print(f"    Trainable tokens (mask_loss=1): {trainable_tokens}")

        if trainable_tokens == 0:
            debug_print(f"    ⚠️  WARNING: No trainable tokens! Loss will be 0!")
        elif trainable_tokens < 5:
            debug_print(f"    ⚠️  WARNING: Very few trainable tokens! Loss might be noisy!")


def check_gradients(grads: Dict, step: int = 0):
    """
    Check 6: Gradient Flow Analysis

    Verify that gradients are flowing to all parts of the model.
    """
    if not _DEBUG_ENABLED:
        return

    global _CHECK_COUNTER
    if _CHECK_COUNTER % 100 != 1 and step > 10:
        return

    print_section(f"Gradient Flow Analysis (Step {step})")

    def compute_norm(x):
        if x is None:
            return 0.0
        return float(jnp.sqrt(jnp.sum(x**2)))

    # Compute gradient norms for each component
    components = {}

    if 'img' in grads:
        img_grads = grads['img']
        img_norms = jax.tree_util.tree_map(compute_norm, img_grads)
        img_leaves = jax.tree_util.tree_leaves(img_norms)
        components['img'] = {
            'max': max(img_leaves) if img_leaves else 0.0,
            'min': min(img_leaves) if img_leaves else 0.0,
            'mean': sum(img_leaves) / len(img_leaves) if img_leaves else 0.0,
            'count': len(img_leaves)
        }

    if 'llm' in grads:
        llm_grads = grads['llm']
        llm_norms = jax.tree_util.tree_map(compute_norm, llm_grads)
        llm_leaves = jax.tree_util.tree_leaves(llm_norms)
        components['llm'] = {
            'max': max(llm_leaves) if llm_leaves else 0.0,
            'min': min(llm_leaves) if llm_leaves else 0.0,
            'mean': sum(llm_leaves) / len(llm_leaves) if llm_leaves else 0.0,
            'count': len(llm_leaves)
        }

    # Print results
    for component, stats in components.items():
        debug_print(f"\n{component.upper()} Gradients:")
        debug_print(f"  Number of parameters: {stats['count']}")
        debug_print(f"  Max gradient norm: {stats['max']:.6f}")
        debug_print(f"  Min gradient norm: {stats['min']:.6f}")
        debug_print(f"  Mean gradient norm: {stats['mean']:.6f}")

        # Check for issues
        if stats['max'] < 1e-8:
            debug_print(f"  🚨 CRITICAL: {component.upper()} gradients are effectively ZERO!")
            debug_print(f"  🚨 {component.upper()} is NOT being trained!")
        elif stats['max'] < 1e-6:
            debug_print(f"  ⚠️  WARNING: {component.upper()} gradients are very small")
            debug_print(f"     Training might be very slow")
        elif stats['max'] > 100.0:
            debug_print(f"  ⚠️  WARNING: {component.upper()} gradients are very large!")
            debug_print(f"     This might cause instability. Consider gradient clipping")
        else:
            debug_print(f"  ✓ {component.upper()} gradients are in normal range")

    # Compare image vs LLM gradients
    if 'img' in components and 'llm' in components:
        ratio = components['img']['mean'] / (components['llm']['mean'] + 1e-10)
        debug_print(f"\nGradient Ratio (img/llm): {ratio:.4f}")
        if ratio < 0.01:
            debug_print("⚠️  WARNING: Image gradients are much smaller than LLM gradients!")
            debug_print("   Image encoder might not be learning effectively")
        elif ratio > 100:
            debug_print("⚠️  WARNING: Image gradients are much larger than LLM gradients!")
            debug_print("   Consider using differential learning rates")


def check_model_outputs(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask_loss: jnp.ndarray,
    step: int = 0
):
    """
    Check 7: Model Output Analysis

    Analyze model predictions and loss calculation.
    """
    if not _DEBUG_ENABLED:
        return

    global _CHECK_COUNTER
    if _CHECK_COUNTER % 100 != 1 and step > 10:
        return

    print_section(f"Model Output Analysis (Step {step})")

    # Logits analysis
    debug_print("--- Logits ---")
    debug_print(f"Logits shape: {logits.shape}")
    debug_print(f"Logits dtype: {logits.dtype}")
    debug_print(f"Logits range: [{jnp.min(logits):.3f}, {jnp.max(logits):.3f}]")

    # Predictions
    predictions = jnp.argmax(logits, axis=-1)
    debug_print(f"\n--- Predictions ---")

    # Accuracy on trainable tokens
    for i in range(min(2, logits.shape[0])):
        sample_preds = predictions[i]
        sample_targets = targets[i]
        sample_mask = mask_loss[i]

        # Only consider trainable tokens
        trainable_positions = jnp.where(sample_mask > 0)[0]
        if len(trainable_positions) > 0:
            correct = jnp.sum(
                (sample_preds[trainable_positions] == sample_targets[trainable_positions])
            )
            total = len(trainable_positions)
            accuracy = float(correct) / total * 100

            debug_print(f"  Sample {i}:")
            debug_print(f"    Trainable tokens: {total}")
            debug_print(f"    Correct predictions: {correct}")
            debug_print(f"    Accuracy: {accuracy:.1f}%")

            # Show first few predictions
            debug_print(f"    First 5 predictions: {sample_preds[trainable_positions[:5]]}")
            debug_print(f"    First 5 targets:     {sample_targets[trainable_positions[:5]]}")

    # Entropy analysis (measure of confidence)
    probs = jax.nn.softmax(logits, axis=-1)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
    mean_entropy = jnp.mean(entropy)

    debug_print(f"\n--- Prediction Confidence ---")
    debug_print(f"Mean entropy: {mean_entropy:.3f}")
    debug_print(f"Max entropy (uniform): {jnp.log(logits.shape[-1]):.3f}")

    confidence_ratio = 1.0 - (mean_entropy / jnp.log(logits.shape[-1]))
    debug_print(f"Confidence ratio: {confidence_ratio:.1%}")

    if confidence_ratio < 0.1:
        debug_print("⚠️  WARNING: Model is very uncertain (nearly uniform distribution)")
        debug_print("   Model might not be learning anything useful")


def check_attention_mask(
    attn_mask: jnp.ndarray,
    num_img_tokens: int,
    step: int = 0
):
    """
    Check 8: Attention Mask Analysis

    Verify that text tokens can attend to image tokens.
    """
    if not _DEBUG_ENABLED:
        return

    global _CHECK_COUNTER
    if _CHECK_COUNTER % 100 != 1 and step > 10:
        return

    print_section(f"Attention Mask Analysis (Step {step})")

    # Analyze first sample
    mask = attn_mask[0]  # [seq_len, seq_len]
    seq_len = mask.shape[0]

    debug_print(f"Attention mask shape: {attn_mask.shape}")
    debug_print(f"First sample seq_len: {seq_len}")
    debug_print(f"Image tokens: 0-{num_img_tokens-1}")
    debug_print(f"Text tokens: {num_img_tokens}-{seq_len-1}")

    # Check: Can text tokens attend to image tokens?
    if seq_len > num_img_tokens:
        text_to_img_attention = mask[num_img_tokens:, :num_img_tokens]
        can_attend = jnp.any(text_to_img_attention)
        attend_ratio = jnp.mean(text_to_img_attention.astype(jnp.float32))

        debug_print(f"\n--- Text → Image Attention ---")
        debug_print(f"Can text attend to images: {can_attend}")
        debug_print(f"Attention coverage: {attend_ratio:.1%}")

        if not can_attend:
            debug_print("🚨 CRITICAL: Text tokens CANNOT attend to image tokens!")
            debug_print("🚨 This means images are COMPLETELY IGNORED!")
        elif attend_ratio < 0.5:
            debug_print("⚠️  WARNING: Text can only attend to some image tokens")
        else:
            debug_print("✓ Text tokens can attend to image tokens")

    # Check: Can image tokens attend to each other?
    if num_img_tokens > 0:
        img_to_img_attention = mask[:num_img_tokens, :num_img_tokens]
        all_can_attend = jnp.all(img_to_img_attention)

        debug_print(f"\n--- Image → Image Attention ---")
        debug_print(f"All image tokens can attend to each other: {all_can_attend}")

        if not all_can_attend:
            debug_print("⚠️  WARNING: Image tokens cannot fully attend to each other")


def check_loss_value(loss: float, step: int = 0):
    """
    Check 9: Loss Value Analysis

    Track loss progression and detect issues.
    """
    if not _DEBUG_ENABLED:
        return

    global _CHECK_COUNTER
    if _CHECK_COUNTER % 10 == 1 or step < 10:
        debug_print(f"\n[Step {step}] Loss: {loss:.6f}")

        # Check for common issues
        if loss == 0.0:
            debug_print("⚠️  WARNING: Loss is exactly 0! This is usually wrong!")
        elif jnp.isnan(loss):
            debug_print("🚨 CRITICAL: Loss is NaN! Training has diverged!")
        elif jnp.isinf(loss):
            debug_print("🚨 CRITICAL: Loss is Inf! Check for division by zero!")
        elif loss > 10.0:
            debug_print("⚠️  WARNING: Loss is very high! Model might not be learning!")


def run_comprehensive_check(
    batch: Dict,
    params: Dict,
    grads: Dict,
    loss: float,
    step: int = 0,
    logits: Optional[jnp.ndarray] = None,
    attn_mask: Optional[jnp.ndarray] = None,
    learning_rate: Optional[float] = None,
):
    """
    Run all checks in one function.

    Call this in your training loop:

    ```python
    if os.environ.get("DEBUG_TRAINING", "0") == "1":
        from src.debug_utils import run_comprehensive_check
        run_comprehensive_check(
            batch=batch,
            params=params,
            grads=grads,
            loss=loss,
            step=step,
            logits=text_logits,  # optional
            attn_mask=attn_mask,  # optional
            learning_rate=current_lr,  # optional
        )
    ```
    """
    if not _DEBUG_ENABLED:
        return

    print(f"\n\n{'#'*80}")
    print(f"# COMPREHENSIVE DEBUG CHECK - STEP {step}")
    print(f"{'#'*80}\n")

    # Run all checks
    if learning_rate is not None:
        check_learning_rate(learning_rate, step)

    check_batch_data(batch, step)
    check_gradients(grads, step)
    check_loss_value(loss, step)

    if logits is not None and 'text' in batch and 'mask_loss' in batch:
        # Shift for auto-regressive alignment
        targets = batch['text'][:, 1:]
        mask_loss = batch['mask_loss'][:, 1:]
        check_model_outputs(logits, targets, mask_loss, step)

    if attn_mask is not None:
        # Estimate image token count
        # Assume 224x224 image with patch size 14 = 256 tokens per image
        # And batch has 'num_images' field
        if 'num_images' in batch:
            num_images = int(batch['num_images'][0])
            num_img_tokens = num_images * 256  # 256 tokens per image
            check_attention_mask(attn_mask, num_img_tokens, step)

    print(f"\n{'#'*80}")
    print(f"# END OF DEBUG CHECK - STEP {step}")
    print(f"{'#'*80}\n\n")


# Quick test
if __name__ == "__main__":
    print("Debug utilities loaded.")
    print(f"Debug mode: {'ENABLED' if _DEBUG_ENABLED else 'DISABLED'}")
    print(f"Set DEBUG_TRAINING=1 to enable comprehensive debugging.")

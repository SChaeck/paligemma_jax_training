"""
Training utilities for PaliGemma fine-tuning.

This module provides training loop utilities, learning rate schedules,
and metric computation.
"""

import functools
import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import Config


# =============================================================================
# Global precision settings - use bfloat16 for speed (matches big_vision/OpenPi)
# =============================================================================
def setup_bfloat16():
    """Enable bfloat16 as default dtype for better performance."""
    # This affects JAX default operations
    jax.config.update("jax_default_matmul_precision", "bfloat16")


def get_num_devices() -> int:
    """Get number of available devices (GPUs/TPUs)."""
    return jax.device_count()


def print_device_info():
    """Print information about available devices."""
    devices = jax.devices()
    print(f"JAX devices: {len(devices)}")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d.platform}: {d.device_kind}")


def setup_big_vision(config: Config):
    """Add big_vision to path if needed."""
    if config.system.big_vision_path not in sys.path:
        if os.path.exists(config.system.big_vision_path):
            sys.path.insert(0, config.system.big_vision_path)


def create_learning_rate_schedule(
    base_learning_rate: float,
    total_steps: int,
    warmup_percent: float = 0.1,
    schedule_type: str = "cosine",
    config: Config = None,
) -> Callable[[int], float]:
    """
    Create learning rate schedule function.

    Args:
        base_learning_rate: Maximum learning rate
        total_steps: Total number of training steps
        warmup_percent: Percentage of steps for warmup
        schedule_type: Type of schedule ("cosine", "constant", "linear")
        config: Optional config for big_vision path

    Returns:
        Function that takes step and returns learning rate
    """
    # Handle constant schedule directly (not supported by big_vision)
    if schedule_type == "constant":
        warmup_steps = int(total_steps * warmup_percent)
        
        def constant_schedule(step: int) -> float:
            """Constant LR with warmup."""
            if warmup_steps > 0 and step < warmup_steps:
                return base_learning_rate * (step / warmup_steps)
            return base_learning_rate
        
        return constant_schedule
    
    # Use big_vision for other schedule types
    if config:
        setup_big_vision(config)

    import big_vision.utils

    schedule_fn = big_vision.utils.create_learning_rate_schedule(
        total_steps=total_steps + 1,
        base=base_learning_rate,
        decay_type=schedule_type,
        warmup_percent=warmup_percent,
    )

    return schedule_fn


# =============================================================================
# Optimizer utilities (Adam with warmup + cosine decay)
# =============================================================================
def create_optimizer(
    learning_rate: float,
    total_steps: int,
    warmup_percent: float = 0.1,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    """
    Create AdamW optimizer with warmup and cosine decay schedule.

    This matches the optimizer setup used in big_vision and paligemma2 PyTorch.

    Args:
        learning_rate: Peak learning rate
        total_steps: Total number of training steps
        warmup_percent: Percentage of steps for linear warmup
        weight_decay: Weight decay coefficient (L2 regularization)
        max_grad_norm: Maximum gradient norm for clipping
        b1: Adam beta1 parameter
        b2: Adam beta2 parameter
        eps: Adam epsilon parameter

    Returns:
        optax GradientTransformation
    """
    warmup_steps = int(total_steps * warmup_percent)

    # Learning rate schedule: linear warmup + cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.01,  # End at 1% of peak
    )

    # Build optimizer chain
    optimizer_chain = []

    # 1. Gradient clipping
    if max_grad_norm > 0:
        optimizer_chain.append(optax.clip_by_global_norm(max_grad_norm))

    # 2. AdamW optimizer
    optimizer_chain.append(
        optax.adamw(
            learning_rate=schedule,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
        )
    )

    return optax.chain(*optimizer_chain)


def create_optimizer_state(optimizer: optax.GradientTransformation, params: Dict) -> optax.OptState:
    """
    Initialize optimizer state.

    Args:
        optimizer: optax optimizer
        params: Model parameters

    Returns:
        Initialized optimizer state
    """
    return optimizer.init(params)


# JIT-compiled training step with Adam optimizer
@functools.partial(jax.jit, donate_argnums=(0, 1), static_argnames=('model', 'optimizer'))
def compiled_train_step_adam(
    params: Dict,
    opt_state: optax.OptState,
    batch: Dict,
    model: Any,
    optimizer: optax.GradientTransformation,
    trainable_mask: Dict,
) -> Tuple[Dict, optax.OptState, float]:
    """
    JIT-compiled training step with Adam optimizer.

    Args:
        params: Model parameters
        opt_state: Optimizer state
        batch: Training batch
        model: PaliGemma model (static)
        optimizer: optax optimizer (static)
        trainable_mask: Mask of trainable parameters

    Returns:
        Tuple of (new_params, new_opt_state, loss)
    """
    imgs = batch["image"]
    txts = batch["text"]
    mask_ar = batch["mask_ar"]
    # Get num_images for proper masking of padded image tokens
    num_images = batch.get("num_images", None)

    def loss_fn(params):
        text_logits, _ = model.apply(
            {"params": params},
            imgs,
            txts[:, :-1],
            mask_ar[:, :-1],
            num_images=num_images,  # Pass num_images for proper masking
            train=True
        )

        logp = jax.nn.log_softmax(text_logits, axis=-1)
        mask_loss = batch["mask_loss"][:, 1:]
        targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])

        token_pplx = jnp.sum(logp * targets, axis=-1)
        example_loss = -jnp.sum(token_pplx * mask_loss, axis=-1)
        example_loss /= jnp.clip(jnp.sum(mask_loss, -1), 1)

        return jnp.mean(example_loss)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Mask gradients for frozen parameters
    grads = jax.tree.map(
        lambda g, m: jnp.where(m, g, jnp.zeros_like(g)),
        grads,
        trainable_mask
    )

    # Apply optimizer update
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss


@functools.partial(jax.jit, static_argnames=('model',), donate_argnums=())
def compute_loss_and_grads_for_accum(
    params: Dict,
    batch: Dict,
    model: Any,
    trainable_mask: Dict,
) -> Tuple[float, Dict]:
    """
    JIT-compiled loss and gradient computation for gradient accumulation.
    
    OPTIMIZED for memory efficiency:
    - JIT-compiled to prevent memory leaks
    - Returns gradients only (no intermediate activations)
    - Uses inline=True to force memory cleanup

    IMPORTANT: This function is JIT-compiled to prevent memory leaks during
    gradient accumulation. Without JIT, each iteration would accumulate
    intermediate activations in memory.

    Args:
        params: Model parameters (NOT donated - used multiple times)
        batch: Training batch
        model: PaliGemma model (static - will be traced once)
        trainable_mask: Mask of trainable parameters

    Returns:
        Tuple of (loss, masked_gradients)
    """
    imgs = batch["image"]
    txts = batch["text"]
    mask_ar = batch["mask_ar"]
    # Get num_images for proper masking of padded image tokens
    num_images = batch.get("num_images", None)

    def loss_fn(params):
        text_logits, _ = model.apply(
            {"params": params},
            imgs,
            txts[:, :-1],
            mask_ar[:, :-1],
            num_images=num_images,  # Pass num_images for proper masking
            train=True
        )

        logp = jax.nn.log_softmax(text_logits, axis=-1)
        mask_loss = batch["mask_loss"][:, 1:]
        targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])

        # Debug: Log mask_loss to verify EOS padding is excluded from training
        # mask_loss=1 means the token contributes to loss, mask_loss=0 means it's excluded
        # EOS token itself has mask_loss=1, but padding after EOS has mask_loss=0
        jax.debug.print("[LOSS DEBUG] mask_loss shape: {shape}, trainable tokens per sample: {nz}", 
                        shape=mask_loss.shape, nz=jnp.sum(mask_loss, axis=-1))

        token_pplx = jnp.sum(logp * targets, axis=-1)
        example_loss = -jnp.sum(token_pplx * mask_loss, axis=-1)
        example_loss /= jnp.clip(jnp.sum(mask_loss, -1), 1)

        return jnp.mean(example_loss)

    # Use value_and_grad with explicit memory management
    loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(params)

    # Mask gradients for frozen parameters
    grads = jax.tree.map(
        lambda g, m: jnp.where(m, g, jnp.zeros_like(g)),
        grads,
        trainable_mask
    )

    return loss, grads


@jax.jit
def accumulate_gradients(accumulated_grads: Dict, grads: Dict) -> Dict:
    """
    JIT-compiled gradient accumulation.
    
    This function is JIT-compiled to prevent memory leaks during
    gradient accumulation. It efficiently adds gradients in-place.
    
    Args:
        accumulated_grads: Previously accumulated gradients
        grads: New gradients to add
        
    Returns:
        Updated accumulated gradients
    """
    return jax.tree.map(lambda x, y: x + y, accumulated_grads, grads)


@functools.partial(jax.jit, donate_argnums=(0, 1), static_argnames=('optimizer',))
def apply_accumulated_gradients_adam(
    params: Dict,
    opt_state: optax.OptState,
    accumulated_grads: Dict,
    optimizer: optax.GradientTransformation,
    num_accum_steps: int,
) -> Tuple[Dict, optax.OptState]:
    """
    Apply accumulated gradients using Adam optimizer.

    Args:
        params: Model parameters
        opt_state: Optimizer state
        accumulated_grads: Accumulated gradients (sum, not average)
        optimizer: optax optimizer
        num_accum_steps: Number of accumulation steps (to average grads)

    Returns:
        Tuple of (new_params, new_opt_state)
    """
    # Average the accumulated gradients
    avg_grads = jax.tree.map(lambda g: g / num_accum_steps, accumulated_grads)

    # Apply optimizer update
    updates, new_opt_state = optimizer.update(avg_grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state


# JIT-compiled training step (legacy SGD - kept for backward compatibility)
@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=('model',))
def compiled_train_step(
    params: Dict,
    batch: Dict,
    model: Any,
    trainable_mask: Dict,
    learning_rate: float,
    max_grad_norm: float = 1.0,
) -> Tuple[Dict, float]:
    """
    JIT-compiled version of training step.

    Args:
        params: Model parameters
        batch: Training batch
        model: PaliGemma model (static)
        trainable_mask: Mask of trainable parameters
        learning_rate: Learning rate
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Tuple of (new_params, loss)
    """
    imgs = batch["image"]
    txts = batch["text"]
    mask_ar = batch["mask_ar"]
    num_images = batch.get("num_images", None)

    def loss_fn(params):
        text_logits, _ = model.apply(
            {"params": params},
            imgs,
            txts[:, :-1],
            mask_ar[:, :-1],
            num_images=num_images,
            train=True
        )

        logp = jax.nn.log_softmax(text_logits, axis=-1)
        mask_loss = batch["mask_loss"][:, 1:]
        targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])

        token_pplx = jnp.sum(logp * targets, axis=-1)
        example_loss = -jnp.sum(token_pplx * mask_loss, axis=-1)
        example_loss /= jnp.clip(jnp.sum(mask_loss, -1), 1)

        return jnp.mean(example_loss)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Apply gradient clipping (use JAX operations instead of Python if)
    grad_norm = jnp.sqrt(sum(
        jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)
    ))
    # If max_grad_norm <= 0, use 1.0 (no clipping), otherwise clip
    clip_factor = jnp.where(
        max_grad_norm > 0,
        jnp.clip(max_grad_norm / (grad_norm + 1e-8), a_max=1.0),
        1.0
    )
    grads = jax.tree.map(lambda g: g * clip_factor, grads)

    def apply_grad(param, gradient, trainable):
        # Use jnp.where instead of Python if for JAX tracing compatibility
        return jnp.where(trainable, param - learning_rate * gradient, param)

    params = jax.tree.map(apply_grad, params, grads, trainable_mask)

    return params, loss


@functools.partial(jax.jit, static_argnames=('model',))
def compute_loss_and_grads(params: Dict, batch: Dict, model: Any) -> Tuple[float, Dict]:
    """
    JIT-compiled loss and gradient computation for gradient accumulation.

    Args:
        params: Model parameters
        batch: Training batch
        model: PaliGemma model (static)

    Returns:
        Tuple of (loss, gradients)
    """
    imgs = batch["image"]
    txts = batch["text"]
    mask_ar = batch["mask_ar"]
    num_images = batch.get("num_images", None)

    def loss_fn(params):
        text_logits, _ = model.apply(
            {"params": params},
            imgs,
            txts[:, :-1],
            mask_ar[:, :-1],
            num_images=num_images,
            train=True
        )

        logp = jax.nn.log_softmax(text_logits, axis=-1)
        mask_loss = batch["mask_loss"][:, 1:]
        targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])

        token_pplx = jnp.sum(logp * targets, axis=-1)
        example_loss = -jnp.sum(token_pplx * mask_loss, axis=-1)
        example_loss /= jnp.clip(jnp.sum(mask_loss, -1), 1)

        return jnp.mean(example_loss)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads


# =============================================================================
# Multi-GPU (pmap) training step
# =============================================================================
def create_pmap_train_step(model: Any):
    """
    Create a pmap-based training step for multi-GPU training.

    This function creates a parallelized training step that:
    1. Runs forward/backward pass on each device
    2. Averages gradients across devices using pmean
    3. Applies the same update on all devices

    Args:
        model: PaliGemma model (will be static in the returned function)

    Returns:
        pmap'd training step function
    """
    def train_step(params, batch, trainable_mask, learning_rate, max_grad_norm):
        """Single device training step (to be pmap'd)."""
        imgs = batch["image"]
        txts = batch["text"]
        mask_ar = batch["mask_ar"]
        num_images = batch.get("num_images", None)

        def loss_fn(params):
            text_logits, _ = model.apply(
                {"params": params},
                imgs,
                txts[:, :-1],
                mask_ar[:, :-1],
                num_images=num_images,
                train=True
            )

            logp = jax.nn.log_softmax(text_logits, axis=-1)
            mask_loss = batch["mask_loss"][:, 1:]
            targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])

            token_pplx = jnp.sum(logp * targets, axis=-1)
            example_loss = -jnp.sum(token_pplx * mask_loss, axis=-1)
            example_loss /= jnp.clip(jnp.sum(mask_loss, -1), 1)

            return jnp.mean(example_loss)

        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Average gradients across devices
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss = jax.lax.pmean(loss, axis_name="batch")

        # Apply gradient clipping
        grad_norm = jnp.sqrt(sum(
            jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)
        ))
        clip_factor = jnp.where(
            max_grad_norm > 0,
            jnp.clip(max_grad_norm / (grad_norm + 1e-8), a_max=1.0),
            1.0
        )
        grads = jax.tree.map(lambda g: g * clip_factor, grads)

        def apply_grad(param, gradient, trainable):
            return jnp.where(trainable, param - learning_rate * gradient, param)

        params = jax.tree.map(apply_grad, params, grads, trainable_mask)

        return params, loss

    # Create pmap'd version
    return jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))


def create_pmap_train_step_adam(model: Any, optimizer: optax.GradientTransformation):
    """
    Create a pmap-based training step with Adam optimizer for multi-GPU training.

    Args:
        model: PaliGemma model (will be static in the returned function)
        optimizer: optax optimizer (must be static - schedule is handled per-device)

    Returns:
        pmap'd training step function

    Note:
        The optimizer is captured in the closure and used on each device.
        Learning rate schedules are handled by the optimizer's internal state,
        which is properly replicated across devices.
    """
    def train_step(params, opt_state, batch, trainable_mask):
        """Single device training step with Adam (to be pmap'd)."""
        imgs = batch["image"]
        txts = batch["text"]
        mask_ar = batch["mask_ar"]
        num_images = batch.get("num_images", None)

        def loss_fn(params):
            text_logits, _ = model.apply(
                {"params": params},
                imgs,
                txts[:, :-1],
                mask_ar[:, :-1],
                num_images=num_images,
                train=True
            )

            logp = jax.nn.log_softmax(text_logits, axis=-1)
            mask_loss = batch["mask_loss"][:, 1:]
            targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])

            token_pplx = jnp.sum(logp * targets, axis=-1)
            example_loss = -jnp.sum(token_pplx * mask_loss, axis=-1)
            example_loss /= jnp.clip(jnp.sum(mask_loss, -1), 1)

            return jnp.mean(example_loss)

        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Average gradients across devices BEFORE masking
        # This ensures all devices see the same averaged gradients
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss = jax.lax.pmean(loss, axis_name="batch")

        # Mask gradients for frozen parameters
        grads = jax.tree.map(
            lambda g, m: jnp.where(m, g, jnp.zeros_like(g)),
            grads,
            trainable_mask
        )

        # Apply optimizer update
        # Each device updates its own copy with the same averaged gradients
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss

    # Create pmap'd version
    return jax.pmap(train_step, axis_name="batch", donate_argnums=(0, 1))


def replicate_params(params: Dict) -> Dict:
    """Replicate parameters across all devices for pmap."""
    return jax.device_put_replicated(params, jax.devices())


def unreplicate_params(params: Dict) -> Dict:
    """Get parameters from first device (all devices have same params after pmap)."""
    return jax.tree.map(lambda x: x[0], params)


def shard_batch_for_pmap(batch: Dict, num_devices: int) -> Dict:
    """
    Reshape batch for pmap: [batch_size, ...] -> [num_devices, batch_per_device, ...]

    Args:
        batch: Batch dictionary with arrays
        num_devices: Number of devices

    Returns:
        Reshaped batch for pmap
    """
    def reshape_array(x):
        if not isinstance(x, (np.ndarray, jnp.ndarray)):
            return x
        batch_size = x.shape[0]
        if batch_size % num_devices != 0:
            raise ValueError(
                f"Batch size {batch_size} must be divisible by num_devices {num_devices}"
            )
        new_shape = (num_devices, batch_size // num_devices) + x.shape[1:]
        return x.reshape(new_shape)

    return jax.tree.map(reshape_array, batch)


@jax.jit
def apply_gradients(
    params: Dict,
    grads: Dict,
    trainable_mask: Dict,
    learning_rate: float,
    max_grad_norm: float = 1.0,
) -> Dict:
    """
    JIT-compiled gradient application with clipping.

    Args:
        params: Model parameters
        grads: Gradients (already accumulated and averaged)
        trainable_mask: Mask of trainable parameters
        learning_rate: Learning rate
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Updated parameters
    """
    # Apply gradient clipping
    grad_norm = jnp.sqrt(sum(
        jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)
    ))
    clip_factor = jnp.where(
        max_grad_norm > 0,
        jnp.clip(max_grad_norm / (grad_norm + 1e-8), a_max=1.0),
        1.0
    )
    grads = jax.tree.map(lambda g: g * clip_factor, grads)

    def apply_grad(param, gradient, trainable):
        return jnp.where(trainable, param - learning_rate * gradient, param)

    params = jax.tree.map(apply_grad, params, grads, trainable_mask)
    return params


@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=('model',))
def compiled_train_step_with_accumulation(
    params: Dict,
    batches: list,
    model: Any,
    trainable_mask: Dict,
    learning_rate: float,
    max_grad_norm: float = 1.0,
) -> Tuple[Dict, float]:
    """
    Training step with gradient accumulation over multiple batches.

    Args:
        params: Model parameters
        batches: List of training batches to accumulate gradients over
        model: PaliGemma model (static)
        trainable_mask: Mask of trainable parameters
        learning_rate: Learning rate
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Tuple of (new_params, average_loss)
    """
    def loss_fn(params, batch):
        imgs = batch["image"]
        txts = batch["text"]
        mask_ar = batch["mask_ar"]
        num_images = batch.get("num_images", None)
        
        text_logits, _ = model.apply(
            {"params": params},
            imgs,
            txts[:, :-1],
            mask_ar[:, :-1],
            num_images=num_images,
            train=True
        )

        logp = jax.nn.log_softmax(text_logits, axis=-1)
        mask_loss = batch["mask_loss"][:, 1:]
        targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])

        token_pplx = jnp.sum(logp * targets, axis=-1)
        example_loss = -jnp.sum(token_pplx * mask_loss, axis=-1)
        example_loss /= jnp.clip(jnp.sum(mask_loss, -1), 1)

        return jnp.mean(example_loss)

    # Accumulate gradients over all batches
    total_loss = 0.0
    accumulated_grads = None
    
    for batch in batches:
        loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(params, batch)
        total_loss += loss
        
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = jax.tree.map(lambda x, y: x + y, accumulated_grads, grads)
    
    # Average gradients and loss
    num_batches = len(batches)
    accumulated_grads = jax.tree.map(lambda g: g / num_batches, accumulated_grads)
    avg_loss = total_loss / num_batches

    # Apply gradient clipping (use JAX operations instead of Python if)
    grad_norm = jnp.sqrt(sum(
        jnp.sum(jnp.square(g)) for g in jax.tree.leaves(accumulated_grads)
    ))
    # If max_grad_norm <= 0, use 1.0 (no clipping), otherwise clip
    clip_factor = jnp.where(
        max_grad_norm > 0,
        jnp.clip(max_grad_norm / (grad_norm + 1e-8), a_max=1.0),
        1.0
    )
    accumulated_grads = jax.tree.map(lambda g: g * clip_factor, accumulated_grads)

    def apply_grad(param, gradient, trainable):
        return jnp.where(trainable, param - learning_rate * gradient, param)

    params = jax.tree.map(apply_grad, params, accumulated_grads, trainable_mask)

    return params, avg_loss


class MetricsTracker:
    """Track and aggregate training metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics_history = []
        self.current_epoch_metrics = []

    def update(self, metrics: Dict[str, float]):
        """Add metrics from current step."""
        self.current_epoch_metrics.append(metrics)

    def compute_epoch_summary(self) -> Dict[str, float]:
        """Compute summary statistics for current epoch."""
        if not self.current_epoch_metrics:
            return {}

        all_metrics = {}
        for metrics in self.current_epoch_metrics:
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        summary = {}
        for key, values in all_metrics.items():
            values = np.array(values)
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))

        return summary

    def reset_epoch(self):
        """Reset metrics for new epoch."""
        if self.current_epoch_metrics:
            epoch_summary = self.compute_epoch_summary()
            self.metrics_history.append(epoch_summary)
        self.current_epoch_metrics = []

    def get_history(self) -> list:
        """Get full training history."""
        return self.metrics_history


def create_data_sharding(config: Config = None):
    """
    Create data sharding specification for single-device training.
    
    Note: Uses only the first GPU to avoid NCCL rendezvous issues.
    For multi-GPU training, use pmap instead.

    Returns:
        Sharding specification
    """
    # Use only the first device to avoid multi-GPU NCCL issues
    devices = [jax.devices()[0]]
    mesh = jax.sharding.Mesh(devices, ("data",))
    return jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec("data")
    )


def shard_batch(batch: Dict, sharding, config: Config = None) -> Dict:
    """
    Shard a batch across devices.

    Args:
        batch: Batch dictionary
        sharding: Sharding specification
        config: Optional config for big_vision path

    Returns:
        Sharded batch
    """
    if config:
        setup_big_vision(config)

    import big_vision.utils
    return big_vision.utils.reshard(batch, sharding)

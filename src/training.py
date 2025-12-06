"""
Training utilities for PaliGemma fine-tuning.

This module provides training loop utilities, learning rate schedules,
and metric computation.
"""

import functools
import os
import sys
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

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


# JIT-compiled training step
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

    def loss_fn(params):
        text_logits, _ = model.apply(
            {"params": params},
            imgs,
            txts[:, :-1],
            mask_ar[:, :-1],
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

    def loss_fn(params):
        text_logits, _ = model.apply(
            {"params": params},
            imgs,
            txts[:, :-1],
            mask_ar[:, :-1],
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

        def loss_fn(params):
            text_logits, _ = model.apply(
                {"params": params},
                imgs,
                txts[:, :-1],
                mask_ar[:, :-1],
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
        
        text_logits, _ = model.apply(
            {"params": params},
            imgs,
            txts[:, :-1],
            mask_ar[:, :-1],
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
    Create data sharding specification for multi-device training.

    Returns:
        Sharding specification
    """
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
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

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
) -> Tuple[Dict, float]:
    """
    JIT-compiled version of training step.

    Args:
        params: Model parameters
        batch: Training batch
        model: PaliGemma model (static)
        trainable_mask: Mask of trainable parameters
        learning_rate: Learning rate

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

    def apply_grad(param, gradient, trainable):
        # Use jnp.where instead of Python if for JAX tracing compatibility
        return jnp.where(trainable, param - learning_rate * gradient, param)

    params = jax.tree.map(apply_grad, params, grads, trainable_mask)

    return params, loss


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

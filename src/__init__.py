"""
PaliGemma Training Library

Stateless training modules that read all configuration from environment variables.
"""

from .config import Config, load_config
from .model import load_paligemma_model, create_trainable_mask, prepare_params_for_training
from .data import XVRDataset, create_train_iterator, create_eval_iterator, preprocess_image, preprocess_multi_images, preprocess_tokens, postprocess_tokens
from .training import (
    compiled_train_step,
    compiled_train_step_adam,
    compute_loss_and_grads_for_accum,
    apply_accumulated_gradients_adam,
    create_learning_rate_schedule,
    create_optimizer,
    create_optimizer_state,
    create_pmap_train_step_adam,
    MetricsTracker,
    create_data_sharding,
    shard_batch,
)
from .evaluation import evaluate_model

__all__ = [
    # Config
    "Config",
    "load_config",
    # Model
    "load_paligemma_model",
    "create_trainable_mask",
    "prepare_params_for_training",
    # Data
    "XVRDataset",
    "create_train_iterator",
    "create_eval_iterator",
    "preprocess_image",
    "preprocess_multi_images",
    "preprocess_tokens",
    "postprocess_tokens",
    # Training
    "compiled_train_step",
    "compiled_train_step_adam",
    "compute_loss_and_grads_for_accum",
    "apply_accumulated_gradients_adam",
    "create_learning_rate_schedule",
    "create_optimizer",
    "create_optimizer_state",
    "create_pmap_train_step_adam",
    "MetricsTracker",
    "create_data_sharding",
    "shard_batch",
    # Evaluation
    "evaluate_model",
]

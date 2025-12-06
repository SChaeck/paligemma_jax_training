"""
Configuration management for PaliGemma training.

All configuration is loaded from environment variables, making the training
pipeline completely stateless and reproducible.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _get_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


def _get_int(key: str, default: Optional[int] = None) -> Optional[int]:
    """Get integer from environment variable."""
    val = os.getenv(key)
    if val is None or val == "":
        return default
    return int(val)


def _get_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    val = os.getenv(key)
    if val is None or val == "":
        return default
    return float(val)


def _get_str(key: str, default: str = "") -> str:
    """Get string from environment variable."""
    return os.getenv(key, default)


@dataclass
class ModelConfig:
    """Model configuration."""
    checkpoint_path: str = "./checkpoints/pi05_base_paligemma.npz"
    checkpoint_url: Optional[str] = None  # URL to download checkpoint if not exists locally
    tokenizer_path: str = "./assets/paligemma_tokenizer.model"
    kaggle_handle: str = "google/paligemma/jax/paligemma-3b-pt-224"

    llm_variant: str = "gemma_2b"
    vocab_size: int = 257152
    img_size: int = 224
    img_variant: str = "So400m/14"
    img_pool_type: str = "none"
    img_scan: bool = True
    img_dtype_mm: str = "float16"


@dataclass
class TrainingConfig:
    """Training configuration."""
    trainable_params: str = "attention_only"
    # CRITICAL: Use low learning rate for fine-tuning!
    # PyTorch reference uses 2e-6, NOT 0.03!
    # 0.03 is ~15000x too high and will destabilize training.
    learning_rate: float = 2e-6
    batch_size: int = 8
    gradient_accumulation_steps: int = 1  # Effective batch size = batch_size * gradient_accumulation_steps
    num_epochs: int = 10
    warmup_percent: float = 0.10
    lr_schedule: str = "cosine"  # "cosine", "constant", "linear"
    max_grad_norm: float = 1.0
    precision: str = "float32"  # "float32", "bfloat16", "float16"
    seed: int = 42
    max_images: int = 6  # Maximum number of images per sample


@dataclass
class DataConfig:
    """Data configuration."""
    base_dir: str = "/home/suchae/pi_workspace/XVR"
    train_file: str = "train.jsonl"
    valid_file: str = "valid.jsonl"
    eval_file: str = "xvr_eval.jsonl"
    image_dir: str = "images"

    # max_seq_length is for TEXT tokens only (not including image tokens)
    # Image tokens are handled separately by the model:
    #   - 224px image with 14px patches = 16x16 = 256 tokens per image
    #   - 6 images = 1,536 image tokens
    # XVR dataset analysis:
    #   - Max text: 1086 chars (~300 tokens)
    #   - 95th percentile: 959 chars (~280 tokens)
    #   - Average: 645 chars (~180 tokens)
    # 512 tokens provides ~1.5x headroom for the longest samples
    max_seq_length: int = 512
    shuffle_buffer_size: int = 1000
    prompt_prefix: str = "answer en"

    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""
    output_dir: str = "./outputs"
    log_every: int = 10
    eval_every: int = 100
    checkpoint_every: int = 500
    max_checkpoints_to_keep: int = 3

    use_wandb: bool = False
    wandb_project: str = "paligemma-xvr"
    wandb_entity: Optional[str] = None


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    num_examples: Optional[int] = None
    batch_size: int = 4
    sampler: str = "greedy"


@dataclass
class SystemConfig:
    """System configuration."""
    xla_mem_fraction: float = 0.9
    tf_allow_growth: bool = True
    big_vision_path: str = "/home/suchae/pi_workspace/big_vision"


@dataclass
class Config:
    """Complete training configuration."""
    experiment_name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.logging.output_dir, "checkpoints")

    @property
    def log_dir(self) -> str:
        return os.path.join(self.logging.output_dir, "logs")


def load_config(env_file: Optional[str] = None) -> Config:
    """
    Load configuration from environment variables.

    Args:
        env_file: Optional path to .env file. If not provided, uses .env in current directory.

    Returns:
        Config object with all settings loaded from environment.
    """
    # Load .env file if provided
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        load_dotenv(override=True)

    # Build config from environment
    config = Config(
        experiment_name=_get_str("EXPERIMENT_NAME", "default"),

        model=ModelConfig(
            checkpoint_path=_get_str("MODEL_CHECKPOINT_PATH", "./checkpoints/pi05_base_paligemma.npz"),
            checkpoint_url=_get_str("MODEL_CHECKPOINT_URL", "") or None,
            tokenizer_path=_get_str("MODEL_TOKENIZER_PATH", "./assets/paligemma_tokenizer.model"),
            kaggle_handle=_get_str("MODEL_KAGGLE_HANDLE", "google/paligemma/jax/paligemma-3b-pt-224"),
            llm_variant=_get_str("MODEL_LLM_VARIANT", "gemma_2b"),
            vocab_size=_get_int("MODEL_VOCAB_SIZE", 257152),
            img_size=_get_int("MODEL_IMG_SIZE", 224),
            img_variant=_get_str("MODEL_IMG_VARIANT", "So400m/14"),
            img_pool_type=_get_str("MODEL_IMG_POOL_TYPE", "none"),
            img_scan=_get_bool("MODEL_IMG_SCAN", True),
            img_dtype_mm=_get_str("MODEL_IMG_DTYPE_MM", "float16"),
        ),

        training=TrainingConfig(
            trainable_params=_get_str("TRAINABLE_PARAMS", "attention_only"),
            # CRITICAL: Default to 2e-6 (same as PyTorch reference), NOT 0.03!
            learning_rate=_get_float("LEARNING_RATE", 2e-6),
            batch_size=_get_int("BATCH_SIZE", 8),
            gradient_accumulation_steps=_get_int("GRADIENT_ACCUMULATION_STEPS", 1),
            num_epochs=_get_int("NUM_EPOCHS", 10),
            warmup_percent=_get_float("WARMUP_PERCENT", 0.10),
            lr_schedule=_get_str("LR_SCHEDULE", "cosine"),
            max_grad_norm=_get_float("MAX_GRAD_NORM", 1.0),
            precision=_get_str("PRECISION", "float32"),
            seed=_get_int("SEED", 42),
            max_images=_get_int("MAX_IMAGES", 6),
        ),

        data=DataConfig(
            base_dir=_get_str("DATA_BASE_DIR", "/home/suchae/pi_workspace/XVR"),
            train_file=_get_str("DATA_TRAIN_FILE", "train.jsonl"),
            valid_file=_get_str("DATA_VALID_FILE", "valid.jsonl"),
            eval_file=_get_str("DATA_EVAL_FILE", "xvr_eval.jsonl"),
            image_dir=_get_str("DATA_IMAGE_DIR", "images"),
            max_seq_length=_get_int("MAX_SEQ_LENGTH", 512),
            shuffle_buffer_size=_get_int("SHUFFLE_BUFFER_SIZE", 1000),
            prompt_prefix=_get_str("PROMPT_PREFIX", "answer en"),
            max_train_samples=_get_int("MAX_TRAIN_SAMPLES", None),
            max_eval_samples=_get_int("MAX_EVAL_SAMPLES", None),
        ),

        logging=LoggingConfig(
            output_dir=_get_str("OUTPUT_DIR", "./outputs"),
            log_every=_get_int("LOG_EVERY", 10),
            eval_every=_get_int("EVAL_EVERY", 100),
            checkpoint_every=_get_int("CHECKPOINT_EVERY", 500),
            max_checkpoints_to_keep=_get_int("MAX_CHECKPOINTS_TO_KEEP", 3),
            use_wandb=_get_bool("USE_WANDB", False),
            wandb_project=_get_str("WANDB_PROJECT", "paligemma-xvr"),
            wandb_entity=_get_str("WANDB_ENTITY", "") or None,
        ),

        eval=EvalConfig(
            num_examples=_get_int("EVAL_NUM_EXAMPLES", None),
            batch_size=_get_int("EVAL_BATCH_SIZE", 4),
            sampler=_get_str("EVAL_SAMPLER", "greedy"),
        ),

        system=SystemConfig(
            xla_mem_fraction=_get_float("XLA_MEM_FRACTION", 0.9),
            tf_allow_growth=_get_bool("TF_ALLOW_GROWTH", True),
            big_vision_path=_get_str("BIG_VISION_PATH", "/home/suchae/pi_workspace/big_vision"),
        ),
    )

    return config


def validate_config(config: Config) -> None:
    """
    Validate configuration and raise errors for invalid settings.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Check data directory exists
    if not os.path.exists(config.data.base_dir):
        raise ValueError(f"Data directory not found: {config.data.base_dir}")

    # Check training parameters
    if config.training.batch_size < 1:
        raise ValueError("Batch size must be >= 1")

    if config.training.learning_rate <= 0:
        raise ValueError("Learning rate must be > 0")

    if config.training.num_epochs < 1:
        raise ValueError("Number of epochs must be >= 1")

    # Check data parameters
    if config.data.max_seq_length < 32:
        raise ValueError("Sequence length must be >= 32")

    # Check trainable params option
    valid_options = ["attention_only", "full_llm", "full_model"]
    if config.training.trainable_params not in valid_options:
        raise ValueError(
            f"trainable_params must be one of {valid_options}, "
            f"got {config.training.trainable_params}"
        )

    print("Configuration validated successfully")


def print_config(config: Config) -> None:
    """Print configuration summary."""
    print("\n" + "=" * 80)
    print(f"Experiment: {config.experiment_name}")
    print("=" * 80)

    print(f"\nModel:")
    print(f"  Checkpoint: {config.model.checkpoint_path}")
    print(f"  LLM Variant: {config.model.llm_variant}")
    print(f"  Image Size: {config.model.img_size}")

    print(f"\nTraining:")
    print(f"  Trainable: {config.training.trainable_params}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  LR Schedule: {config.training.lr_schedule}")

    print(f"\nData:")
    print(f"  Base Dir: {config.data.base_dir}")
    print(f"  Train File: {config.data.train_file}")
    print(f"  Max Seq Length: {config.data.max_seq_length}")
    if config.data.max_train_samples:
        print(f"  Max Train Samples: {config.data.max_train_samples}")

    print(f"\nLogging:")
    print(f"  Output Dir: {config.logging.output_dir}")
    print(f"  Log Every: {config.logging.log_every} steps")
    print(f"  Eval Every: {config.logging.eval_every} steps")
    print(f"  W&B: {config.logging.use_wandb}")

    print("=" * 80 + "\n")

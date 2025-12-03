"""
Evaluation utilities for PaliGemma fine-tuning.

This module provides functions for model evaluation and metrics computation.
"""

from typing import Dict, Any, Optional
import numpy as np
import jax

from .config import Config
from .data import XVRDataset, create_eval_iterator, postprocess_tokens
from .training import create_data_sharding, shard_batch


def evaluate_model(
    model: Any,
    params: Dict,
    decode_fn: Any,
    tokenizer: Any,
    eval_dataset: XVRDataset,
    config: Config,
    num_examples: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run evaluation on validation set.

    Args:
        model: PaliGemma model
        params: Model parameters
        decode_fn: Decode function
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        config: Configuration
        num_examples: Override number of examples (None = use config)
        verbose: Whether to print progress

    Returns:
        Dictionary with evaluation metrics
    """
    if verbose:
        print("\nRunning evaluation...")

    num_eval = num_examples or config.eval.num_examples

    eval_iterator = create_eval_iterator(
        eval_dataset,
        batch_size=config.eval.batch_size,
        prompt_prefix=config.data.prompt_prefix,
        num_examples=num_eval,
    )

    predictions = []
    ground_truths = []
    data_sharding = create_data_sharding(config)

    for batch in eval_iterator:
        # Convert to batch and shard
        batch_dict = {k: np.stack([v]) for k, v in batch.items() if k not in ['sample_id', 'ground_truth']}
        # Add _mask key required by big_vision decode_fn
        batch_dict["_mask"] = np.ones(batch_dict["image"].shape[0], dtype=bool)
        batch_dict = shard_batch(batch_dict, data_sharding, config)

        # Generate prediction (limit length to avoid infinite generation)
        max_gen_len = min(32, config.data.max_seq_length)
        tokens = decode_fn(
            {"params": params},
            batch=batch_dict,
            max_decode_len=max_gen_len,
            sampler=config.eval.sampler,
        )

        # Decode tokens
        tokens = jax.device_get(tokens)[0]
        pred_text = postprocess_tokens(tokens, tokenizer)

        # Remove prefix from prediction
        if pred_text.startswith(config.data.prompt_prefix):
            pred_text = pred_text[len(config.data.prompt_prefix):].strip()

        predictions.append(pred_text)
        ground_truths.append(batch['ground_truth'])

    # Compute accuracy (exact match)
    correct_exact = sum(
        p.lower().strip() == g.lower().strip()
        for p, g in zip(predictions, ground_truths)
    )

    # Compute "contains" accuracy (for overfit test - checks if answer is in prediction)
    correct_contains = sum(
        g.lower().strip() in p.lower()
        for p, g in zip(predictions, ground_truths)
    )

    accuracy = correct_exact / len(predictions) if predictions else 0.0
    accuracy_contains = correct_contains / len(predictions) if predictions else 0.0

    if verbose:
        print(f"  Exact Accuracy: {accuracy:.2%} ({correct_exact}/{len(predictions)})")
        print(f"  Contains Accuracy: {accuracy_contains:.2%} ({correct_contains}/{len(predictions)})")

        # Print some examples
        print("\n  Example predictions:")
        for i in range(min(3, len(predictions))):
            print(f"    Pred: {predictions[i]}")
            print(f"    True: {ground_truths[i]}")
            print()

    return {
        "accuracy": accuracy,
        "accuracy_contains": accuracy_contains,
        "correct": correct_exact,
        "correct_contains": correct_contains,
        "total": len(predictions),
        "predictions": predictions,
        "ground_truths": ground_truths,
    }


def compute_metrics(
    predictions: list,
    ground_truths: list,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers

    Returns:
        Dictionary of metrics
    """
    if not predictions or not ground_truths:
        return {"accuracy": 0.0}

    correct = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        pred = pred.strip().lower()
        gt = gt.strip().lower()

        if pred == gt:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

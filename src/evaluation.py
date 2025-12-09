"""
Evaluation utilities for PaliGemma fine-tuning.

This module provides functions for model evaluation and metrics computation.
"""

import json
import os
from typing import Dict, Any, Optional, List
import numpy as np
import jax
from PIL import Image
from tqdm import tqdm

from .config import Config
from .data import XVRDataset, create_eval_iterator, postprocess_tokens, collate_eval_batch
from .training import create_data_sharding, shard_batch


def save_debug_images(image_array: np.ndarray, sample_id: str, base_dir: str) -> List[str]:
    """
    Save preprocessed images to a sample-specific directory for debugging.

    Creates directory structure:
        base_dir/
            sample_id/
                img01.png  (Image 1 - first in model input)
                img02.png  (Image 2)
                ...

    Args:
        image_array: Image in range [-1, 1] with shape [H, W, 3] or [N, H, W, 3]
        sample_id: Sample identifier (used as directory name)
        base_dir: Base directory for all debug images

    Returns:
        List of saved image paths
    """
    # Create sample directory
    sample_dir = os.path.join(base_dir, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    saved_paths = []

    if image_array.ndim == 4:
        # Multi-image: [N, H, W, 3]
        num_images = image_array.shape[0]
        for i in range(num_images):
            img_data = ((image_array[i] + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_data)
            # 1-indexed to match "Image 1", "Image 2" in prompt
            img_path = os.path.join(sample_dir, f"img{i+1:02d}.png")
            img.save(img_path)
            saved_paths.append(img_path)
    else:
        # Single image: [H, W, 3]
        img_data = ((image_array + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_data)
        img_path = os.path.join(sample_dir, "img01.png")
        img.save(img_path)
        saved_paths.append(img_path)

    return saved_paths


def evaluate_model(
    model: Any,
    params: Dict,
    decode_fn: Any,
    tokenizer: Any,
    eval_dataset: XVRDataset,
    config: Config,
    num_examples: Optional[int] = None,
    verbose: bool = True,
    save_results_path: Optional[str] = None,
    max_saved_samples: int = 100,
) -> Dict[str, Any]:
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
        save_results_path: Optional path to save detailed results (JSON)
        max_saved_samples: Maximum number of sample results to save (default: 100)

    Returns:
        Dictionary with evaluation metrics and sample results
    """
    if verbose:
        print("\nRunning evaluation...")

    num_eval = num_examples or config.eval.num_examples

    # Get max_images from config if available
    max_images = getattr(config.training, 'max_images', 6)

    eval_iterator = create_eval_iterator(
        eval_dataset,
        batch_size=config.eval.batch_size,
        prompt_prefix=config.data.prompt_prefix,
        num_examples=num_eval,
        max_images=max_images,
        use_image_grid=config.data.use_image_grid,
        grid_rows=config.data.grid_rows,
        grid_cols=config.data.grid_cols,
    )

    predictions = []
    ground_truths = []
    all_sample_ids = []  # Track all sample IDs
    all_input_prompts = []  # Track input prompts for debugging
    all_num_images = []  # Track number of images per sample
    all_images = []  # Store images for debugging
    data_sharding = create_data_sharding(config)

    # Collect samples into batches
    batch_samples = []
    batch_sample_ids = []
    batch_ground_truths = []
    batch_input_prompts = []
    batch_num_images = []
    batch_images_for_debug = []  # Store images for debugging

    # Calculate total batches for progress bar
    total_samples = num_eval if num_eval else eval_dataset.num_samples
    total_batches = (total_samples + config.eval.batch_size - 1) // config.eval.batch_size

    # Wrap iterator with tqdm
    pbar = tqdm(eval_iterator, total=total_samples, desc="Evaluating", disable=not verbose)

    for sample in pbar:
        batch_samples.append(sample)
        batch_sample_ids.append(sample.get('sample_id', None))
        batch_ground_truths.append(sample.get('ground_truth', None))
        batch_input_prompts.append(sample.get('input_prompt', ''))
        batch_num_images.append(sample.get('num_images', 1))
        batch_images_for_debug.append(sample.get('image', None))

        # Process when we have a full batch or at the end
        if len(batch_samples) >= config.eval.batch_size:
            # Use proper collation for multi-image handling
            collated = collate_eval_batch(
                batch_samples,
                max_images=max_images,
                image_size=config.model.img_size,
            )

            # Create batch dict for model (exclude metadata)
            batch_dict = {
                'image': collated['image'],
                'text': collated['text'],
                'mask_ar': collated['mask_ar'],
                'mask_input': collated['mask_input'],
            }
            # Add _mask key required by big_vision decode_fn
            batch_dict["_mask"] = np.ones(len(batch_samples), dtype=bool)
            batch_dict = shard_batch(batch_dict, data_sharding, config)

            # Generate predictions (limit length to avoid infinite generation)
            max_gen_len = min(32, config.data.max_seq_length)
            tokens = decode_fn(
                {"params": params},
                batch=batch_dict,
                max_decode_len=max_gen_len,
                sampler=config.eval.sampler,
            )

            # Decode tokens for each sample in batch
            tokens = jax.device_get(tokens)
            for i, token_seq in enumerate(tokens):
                pred_text = postprocess_tokens(token_seq, tokenizer)
                # Note: decode_fn returns only generated tokens, not input prefix
                predictions.append(pred_text.strip())
                ground_truths.append(batch_ground_truths[i])
                all_sample_ids.append(batch_sample_ids[i])
                all_input_prompts.append(batch_input_prompts[i])
                all_num_images.append(batch_num_images[i])
                all_images.append(batch_images_for_debug[i])

            # Update progress bar with current accuracy
            if predictions:
                current_acc = sum(p.lower().strip() == g.lower().strip()
                                 for p, g in zip(predictions, ground_truths)) / len(predictions)
                pbar.set_postfix({"acc": f"{current_acc:.2%}"})

            # Reset batch
            batch_samples = []
            batch_sample_ids = []
            batch_ground_truths = []
            batch_input_prompts = []
            batch_num_images = []
            batch_images_for_debug = []

    # Process remaining samples if any
    if batch_samples:
        # Use proper collation for multi-image handling
        collated = collate_eval_batch(
            batch_samples,
            max_images=max_images,
            image_size=config.model.img_size,
        )

        # Create batch dict for model (exclude metadata)
        batch_dict = {
            'image': collated['image'],
            'text': collated['text'],
            'mask_ar': collated['mask_ar'],
            'mask_input': collated['mask_input'],
        }
        batch_dict["_mask"] = np.ones(len(batch_samples), dtype=bool)
        batch_dict = shard_batch(batch_dict, data_sharding, config)

        max_gen_len = min(32, config.data.max_seq_length)
        tokens = decode_fn(
            {"params": params},
            batch=batch_dict,
            max_decode_len=max_gen_len,
            sampler=config.eval.sampler,
        )

        tokens = jax.device_get(tokens)
        for i, token_seq in enumerate(tokens):
            pred_text = postprocess_tokens(token_seq, tokenizer)
            # Note: decode_fn returns only generated tokens, not input prefix
            predictions.append(pred_text.strip())
            ground_truths.append(batch_ground_truths[i])
            all_sample_ids.append(batch_sample_ids[i])
            all_input_prompts.append(batch_input_prompts[i])
            all_num_images.append(batch_num_images[i])
            all_images.append(batch_images_for_debug[i])

        # Update progress bar with current accuracy
        if predictions:
            current_acc = sum(p.lower().strip() == g.lower().strip()
                             for p, g in zip(predictions, ground_truths)) / len(predictions)
            pbar.set_postfix({"acc": f"{current_acc:.2%}"})

    # Close progress bar
    pbar.close()

    # Build per-sample results
    sample_results = []
    for i in range(len(predictions)):
        pred = predictions[i]
        gt = ground_truths[i]
        is_correct = pred.lower().strip() == gt.lower().strip()
        is_contains = gt.lower().strip() in pred.lower()
        sample_results.append({
            "index": i,
            "sample_id": all_sample_ids[i],
            "input_prompt": all_input_prompts[i],
            "num_images": all_num_images[i],
            "prediction": pred,
            "ground_truth": gt,
            "correct_exact": is_correct,
            "correct_contains": is_contains,
        })

    # Compute accuracy (exact match)
    correct_exact = sum(1 for r in sample_results if r["correct_exact"])

    # Compute "contains" accuracy (for overfit test - checks if answer is in prediction)
    correct_contains = sum(1 for r in sample_results if r["correct_contains"])

    accuracy = correct_exact / len(predictions) if predictions else 0.0
    accuracy_contains = correct_contains / len(predictions) if predictions else 0.0

    if verbose:
        print(f"  Exact Accuracy: {accuracy:.2%} ({correct_exact}/{len(predictions)})")
        print(f"  Contains Accuracy: {accuracy_contains:.2%} ({correct_contains}/{len(predictions)})")

        # Print some examples
        print("\n  Example predictions:")
        for i in range(min(3, len(predictions))):
            print(f"    Input: {all_input_prompts[i][:100]}...")
            print(f"    Pred: {predictions[i]}")
            print(f"    True: {ground_truths[i]}")
            print(f"    Num Images: {all_num_images[i]}")
            print()

    # Save sample results to file if path provided
    if save_results_path:
        # Limit number of saved samples
        samples_to_save = sample_results[:max_saved_samples]

        # Ensure directory exists
        results_dir = os.path.dirname(save_results_path)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)

        # Create images subdirectory for debug images
        images_dir = os.path.join(results_dir, "debug_images") if results_dir else "debug_images"
        os.makedirs(images_dir, exist_ok=True)

        # Save debug images and add image paths to samples
        for i, sample in enumerate(samples_to_save):
            sample_idx = sample["index"]
            if sample_idx < len(all_images) and all_images[sample_idx] is not None:
                try:
                    saved_paths = save_debug_images(
                        all_images[sample_idx],
                        sample_id=sample['sample_id'],
                        base_dir=images_dir,
                    )
                    # Store list of individual image paths
                    sample["debug_image_paths"] = saved_paths
                    sample["debug_image_count"] = len(saved_paths)
                except Exception as e:
                    sample["debug_image_paths"] = [f"Error saving image: {e}"]
                    sample["debug_image_count"] = 0

        results_data = {
            "summary": {
                "accuracy": accuracy,
                "accuracy_contains": accuracy_contains,
                "correct_exact": correct_exact,
                "correct_contains": correct_contains,
                "total": len(predictions),
                "saved_samples": len(samples_to_save),
            },
            "samples": samples_to_save,
        }

        with open(save_results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"  Saved {len(samples_to_save)} sample results to: {save_results_path}")
            print(f"  Debug images saved to: {images_dir}")

    return {
        "accuracy": accuracy,
        "accuracy_contains": accuracy_contains,
        "correct": correct_exact,
        "correct_contains": correct_contains,
        "total": len(predictions),
        "predictions": predictions,
        "ground_truths": ground_truths,
        "sample_results": sample_results[:max_saved_samples],
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

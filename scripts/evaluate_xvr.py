#!/usr/bin/env python3
"""
XVR External Dataset Evaluation Script (JAX Version)

This script evaluates a trained JAX model checkpoint on external XVR datasets.
It uses the same data processing pipeline as training/validation for consistency.

Usage:
    python scripts/evaluate_xvr.py \
        --checkpoint outputs/checkpoints/checkpoint_001000.npz \
        --data_path /path/to/test_data.jsonl \
        --out_dir outputs/xvr_eval_results
"""

import os
import sys
import json
import argparse
import ast
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, Optional, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress TensorFlow warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import jax
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

from src.config import load_config, print_config
from src.model import load_paligemma_model, create_trainable_mask, prepare_params_for_training, setup_big_vision
from src.data import XVRDataset, preprocess_multi_images, preprocess_tokens, postprocess_tokens
from src.training import create_data_sharding, shard_batch


def load_checkpoint_into_params(checkpoint_path: str, params: Dict, config: Any) -> tuple:
    """
    Load checkpoint and merge into existing params structure.

    Args:
        checkpoint_path: Path to .npz checkpoint file
        params: Existing params structure from model
        config: Configuration (needed for big_vision setup)

    Returns:
        Tuple of (updated_params, step)
    """
    setup_big_vision(config)
    import big_vision.utils

    print(f"Loading checkpoint from {checkpoint_path}...")
    loaded = np.load(checkpoint_path)

    # Extract step from filename if possible
    step = 0
    try:
        import re
        match = re.search(r'checkpoint_(\d+)\.npz', checkpoint_path)
        if match:
            step = int(match.group(1))
    except:
        pass

    # Get flattened current params with paths
    flat_params_list = big_vision.utils.tree_flatten_with_names(params)[0]
    current_paths = {path for path, _ in flat_params_list}

    # Build mapping from checkpoint keys to values
    checkpoint_params = {}
    for key in loaded.files:
        # Remove 'params/' prefix if present
        param_key = key.replace('params/', '')
        checkpoint_params[param_key] = loaded[key]

    # Update params using JAX tree_map with path tracking
    def update_param(path_tuple, current_val):
        # Convert path tuple to string path (e.g., ('img', 'head', 'kernel') -> 'img/head/kernel')
        path_str = '/'.join(str(p) for p in path_tuple)
        if path_str in checkpoint_params:
            return checkpoint_params[path_str]
        return current_val

    # Use jax.tree_util to traverse and update
    import jax.tree_util as tree_util

    # Get the flat structure with paths
    leaves, treedef = tree_util.tree_flatten(params)

    # Map checkpoint values back into tree structure
    loaded_count = 0
    new_leaves = []
    for (path, _), leaf in zip(flat_params_list, leaves):
        if path in checkpoint_params:
            new_leaves.append(checkpoint_params[path])
            loaded_count += 1
        else:
            new_leaves.append(leaf)

    print(f"  Loaded {loaded_count} parameter arrays from checkpoint")

    # Reconstruct params tree
    updated_params = tree_util.tree_unflatten(treedef, new_leaves)

    return updated_params, step


def print_banner(text: str):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def save_debug_image(image_array: np.ndarray, save_path: str) -> None:
    """Save a preprocessed image array to file for debugging."""
    img_data = ((image_array + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_data)
    img.save(save_path)


def normalize_answer(answer: Any) -> str:
    """Normalize answer for comparison (case-insensitive, stripped)."""
    if answer is None:
        return ""
    return str(answer).lower().strip()


def normalize_list_answer(answer: Any) -> List[str]:
    """Normalize list answer for comparison."""
    if isinstance(answer, list):
        return [normalize_answer(item) for item in answer]
    return [normalize_answer(answer)]


def parse_list_answer(answer_str: str) -> List[str]:
    """Parse a string representation of a list into a list."""
    try:
        parsed = ast.literal_eval(answer_str)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (ValueError, SyntaxError):
        pass
    return [answer_str]


def calculate_scores(prediction: str, ground_truth: str, answer_format: str = 'single_choice'):
    """
    Calculate exact match and partial credit scores.

    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        answer_format: 'single_choice' or 'list'

    Returns:
        Tuple of (exact_match: bool, partial_score: float)
    """
    if answer_format == 'list':
        # Parse lists
        pred_list = normalize_list_answer(parse_list_answer(prediction))
        gt_list = normalize_list_answer(parse_list_answer(ground_truth))

        # Exact match
        exact_match = pred_list == gt_list

        # Partial credit
        if len(pred_list) == len(gt_list) and len(gt_list) > 0:
            correct = sum(p == g for p, g in zip(pred_list, gt_list))
            partial_score = correct / len(gt_list)
        else:
            partial_score = 0.0
    else:
        # Single choice - normalize and compare
        pred_norm = normalize_answer(prediction)
        gt_norm = normalize_answer(ground_truth)

        # Remove common prefixes like "image"
        pred_norm = pred_norm.replace("image", "").strip()
        gt_norm = gt_norm.replace("image", "").strip()

        exact_match = pred_norm == gt_norm
        partial_score = 1.0 if exact_match else 0.0

    return exact_match, partial_score


def evaluate_external_dataset(
    model: Any,
    params: Dict,
    decode_fn: Any,
    tokenizer: Any,
    data_path: str,
    config: Any,
    out_dir: str,
    batch_size: int = 8,
    max_new_tokens: int = 32,
    max_saved_samples: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate model on external XVR dataset.

    Args:
        model: PaliGemma model
        params: Model parameters
        decode_fn: Decode function
        tokenizer: Tokenizer
        data_path: Path to JSONL test data
        config: Configuration
        out_dir: Output directory for results
        batch_size: Batch size for inference
        max_new_tokens: Maximum tokens to generate
        max_saved_samples: Maximum samples to save for debugging
        verbose: Print progress

    Returns:
        Dictionary with evaluation metrics
    """
    if verbose:
        print(f"\nEvaluating on: {data_path}")

    # Create dataset - use same preprocessing as training
    data_dir = os.path.dirname(data_path)

    eval_dataset = XVRDataset(
        jsonl_path=data_path,
        image_base_dir=data_dir,
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        image_size=config.model.img_size,
        shuffle_buffer_size=1000,
    )

    if verbose:
        print(f"  Total samples: {eval_dataset.num_samples}")

    # Load all samples
    tf_dataset = eval_dataset.get_tfdata(shuffle=False, repeat=False)

    # Tracking
    all_results = []
    task_stats = defaultdict(lambda: {
        'total': 0,
        'exact_correct': 0,
        'partial_score_sum': 0.0
    })

    data_sharding = create_data_sharding(config)
    prompt_prefix = config.data.prompt_prefix

    # Process samples
    samples_processed = 0
    batch_samples = []
    batch_metadata = []

    iterator = tqdm(tf_dataset.as_numpy_iterator(),
                    total=eval_dataset.num_samples,
                    desc="Evaluating") if verbose else tf_dataset.as_numpy_iterator()

    for line in iterator:
        # Parse sample
        sample = eval_dataset.parse_sample(line.decode('utf-8'))

        if not sample['images']:
            continue

        # Load and process images (same as training)
        try:
            loaded_images = []
            for img_path in sample['images']:
                img = eval_dataset.load_image(img_path)
                loaded_images.append(img)

            image = preprocess_multi_images(loaded_images, size=eval_dataset.image_size)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load images for {sample['sample_id']}: {e}")
            continue

        # Create input (same format as training)
        full_prefix = f"{prompt_prefix} {sample['prompt']}"

        tokens, mask_ar, _, mask_input = preprocess_tokens(
            prefix=full_prefix,
            suffix=None,
            seqlen=eval_dataset.max_seq_length,
            tokenizer=tokenizer,
        )

        # Store for batch processing
        batch_samples.append({
            'image': np.asarray(image),
            'text': np.asarray(tokens),
            'mask_ar': np.asarray(mask_ar),
            'mask_input': np.asarray(mask_input),
        })

        # Get original data for metadata
        original_data = json.loads(line.decode('utf-8'))
        batch_metadata.append({
            'sample_id': sample['sample_id'],
            'task': sample['task'],
            'ground_truth': sample['answer'],
            'input_prompt': full_prefix,
            'num_images': len(loaded_images),
            'answer_format': original_data.get('answer_format', 'single_choice'),
            'image_array': image,  # For debug saving
        })

        # Process batch
        if len(batch_samples) >= batch_size:
            _process_batch(
                batch_samples, batch_metadata, model, params, decode_fn,
                tokenizer, config, data_sharding, max_new_tokens,
                all_results, task_stats
            )
            batch_samples = []
            batch_metadata = []

        samples_processed += 1

    # Process remaining samples
    if batch_samples:
        _process_batch(
            batch_samples, batch_metadata, model, params, decode_fn,
            tokenizer, config, data_sharding, max_new_tokens,
            all_results, task_stats
        )

    # Calculate overall metrics
    total_samples = sum(stats['total'] for stats in task_stats.values())
    total_exact = sum(stats['exact_correct'] for stats in task_stats.values())
    total_partial = sum(stats['partial_score_sum'] for stats in task_stats.values())

    exact_accuracy = total_exact / total_samples if total_samples > 0 else 0
    partial_score = total_partial / total_samples if total_samples > 0 else 0

    # Print results
    if verbose:
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total Samples: {total_samples}")
        print(f"Exact Match Accuracy: {exact_accuracy*100:.2f}%")
        print(f"Partial Credit Score: {partial_score*100:.2f}%")
        print(f"\n--- Performance by Task ---")
        for task, stats in sorted(task_stats.items()):
            exact_acc = stats['exact_correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            partial_acc = stats['partial_score_sum'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {task}:")
            print(f"    Exact Match: {exact_acc:.2f}% ({stats['exact_correct']}/{stats['total']})")
            print(f"    Partial Credit: {partial_acc:.2f}%")

    # Save results
    os.makedirs(out_dir, exist_ok=True)

    # Save debug images
    images_dir = os.path.join(out_dir, "debug_images")
    os.makedirs(images_dir, exist_ok=True)

    samples_to_save = all_results[:max_saved_samples]
    for result in samples_to_save:
        if 'image_array' in result and result['image_array'] is not None:
            img_path = os.path.join(images_dir, f"sample_{result['sample_id']}.png")
            try:
                save_debug_image(result['image_array'], img_path)
                result['debug_image_path'] = img_path
            except Exception as e:
                result['debug_image_path'] = f"Error: {e}"
            # Remove array from results (not JSON serializable)
            del result['image_array']
        elif 'image_array' in result:
            del result['image_array']

    # Remove image_array from all results for JSON serialization
    for result in all_results[max_saved_samples:]:
        if 'image_array' in result:
            del result['image_array']

    # Save detailed results
    results_data = {
        "summary": {
            "data_path": data_path,
            "total_samples": total_samples,
            "exact_match_accuracy": exact_accuracy,
            "partial_credit_score": partial_score,
        },
        "task_stats": {task: dict(stats) for task, stats in task_stats.items()},
        "samples": samples_to_save,
    }

    results_path = os.path.join(out_dir, "eval_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    # Save full predictions
    predictions_path = os.path.join(out_dir, "all_predictions.jsonl")
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Save summary
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("XVR Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data Path: {data_path}\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Exact Match Accuracy: {exact_accuracy*100:.2f}%\n")
        f.write(f"Partial Credit Score: {partial_score*100:.2f}%\n")
        f.write(f"\nPerformance by Task:\n")
        for task, stats in sorted(task_stats.items()):
            exact_acc = stats['exact_correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            partial_acc = stats['partial_score_sum'] / stats['total'] * 100 if stats['total'] > 0 else 0
            f.write(f"  {task}:\n")
            f.write(f"    Exact Match: {exact_acc:.2f}% ({stats['exact_correct']}/{stats['total']})\n")
            f.write(f"    Partial Credit: {partial_acc:.2f}%\n")

    if verbose:
        print(f"\nResults saved to: {out_dir}")
        print(f"  - eval_results.json (detailed with {len(samples_to_save)} samples)")
        print(f"  - all_predictions.jsonl (all predictions)")
        print(f"  - summary.txt")
        print(f"  - debug_images/ ({len(samples_to_save)} images)")

    return {
        "exact_match_accuracy": exact_accuracy,
        "partial_credit_score": partial_score,
        "total_samples": total_samples,
        "task_stats": dict(task_stats),
        "results": all_results,
    }


def _process_batch(
    batch_samples, batch_metadata, model, params, decode_fn,
    tokenizer, config, data_sharding, max_new_tokens,
    all_results, task_stats
):
    """Process a batch of samples."""
    # Create batch dict
    batch_dict = {
        k: np.stack([s[k] for s in batch_samples])
        for k in batch_samples[0].keys()
    }
    batch_dict["_mask"] = np.ones(len(batch_samples), dtype=bool)
    batch_dict = shard_batch(batch_dict, data_sharding, config)

    # Generate predictions
    tokens = decode_fn(
        {"params": params},
        batch=batch_dict,
        max_decode_len=max_new_tokens,
        sampler=config.eval.sampler,
    )

    # Decode and process each prediction
    tokens = jax.device_get(tokens)
    for i, token_seq in enumerate(tokens):
        pred_text = postprocess_tokens(token_seq, tokenizer).strip()
        meta = batch_metadata[i]

        # Calculate scores
        exact_match, partial_score = calculate_scores(
            pred_text,
            meta['ground_truth'],
            meta['answer_format']
        )

        # Update task stats
        task = meta['task']
        task_stats[task]['total'] += 1
        if exact_match:
            task_stats[task]['exact_correct'] += 1
        task_stats[task]['partial_score_sum'] += partial_score

        # Store result
        result = {
            'sample_id': meta['sample_id'],
            'task': meta['task'],
            'input_prompt': meta['input_prompt'],
            'num_images': meta['num_images'],
            'prediction': pred_text,
            'ground_truth': meta['ground_truth'],
            'answer_format': meta['answer_format'],
            'exact_match': exact_match,
            'partial_score': partial_score,
            'image_array': meta.get('image_array'),
        }
        all_results.append(result)


def main():
    parser = argparse.ArgumentParser(description="XVR External Dataset Evaluation (JAX)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file or directory",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to JSONL test data file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for results (default: outputs/xvr_eval/<timestamp>)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=str(PROJECT_ROOT / "envs" / ".env.example"),
        help="Path to .env file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--max_saved_samples",
        type=int,
        default=100,
        help="Maximum samples to save with debug images",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information",
    )

    args = parser.parse_args()

    # Setup output directory
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = str(PROJECT_ROOT / "outputs" / "xvr_eval" / timestamp)

    # Load config
    config = load_config(args.env)

    if args.debug:
        print_config(config)

    print_banner("XVR External Dataset Evaluation")

    # Setup
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config.system.xla_mem_fraction)
    print(f"JAX devices: {jax.devices()}")

    # ==========================================================================
    # Load Model
    # ==========================================================================
    print_banner("Loading Model")

    model, params, tokenizer, decode_fn = load_paligemma_model(config)

    trainable_mask = create_trainable_mask(
        params,
        strategy=config.training.trainable_params,
        config=config,
    )
    params = prepare_params_for_training(params, trainable_mask, config)

    # ==========================================================================
    # Load Checkpoint
    # ==========================================================================
    print_banner("Loading Checkpoint")

    checkpoint_path = args.checkpoint

    # Handle directory path
    if os.path.isdir(checkpoint_path):
        checkpoints = sorted(Path(checkpoint_path).glob("checkpoint_*.npz"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")
        checkpoint_path = str(checkpoints[-1])

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint into params structure
    params, step = load_checkpoint_into_params(checkpoint_path, params, config)
    print(f"  Loaded checkpoint from step {step}")

    # ==========================================================================
    # Run Evaluation
    # ==========================================================================
    print_banner("Running Evaluation")

    metrics = evaluate_external_dataset(
        model=model,
        params=params,
        decode_fn=decode_fn,
        tokenizer=tokenizer,
        data_path=args.data_path,
        config=config,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_saved_samples=args.max_saved_samples,
        verbose=True,
    )

    print_banner("Evaluation Complete")
    print(f"Results saved to: {args.out_dir}")

    return metrics


if __name__ == "__main__":
    main()

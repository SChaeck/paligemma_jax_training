"""
Data loading and preprocessing utilities for XVR dataset.

This module handles loading the XVR dataset in JSONL format and preprocessing
images and text for PaliGemma training.
"""

import json
import os
import queue
import random
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import Config


# =============================================================================
# Data Pipeline Logging
# =============================================================================
# Set DATA_PIPELINE_DEBUG=1 in environment to enable detailed logging
# Or call enable_pipeline_debug() to enable programmatically

_PIPELINE_DEBUG = os.environ.get("DATA_PIPELINE_DEBUG", "0") == "1"
_DEBUG_SAMPLE_COUNT = 0  # Track how many samples we've logged
_DEBUG_MAX_SAMPLES = 3   # Only log first N samples in detail

def enable_pipeline_debug(max_samples: int = 3):
    """Enable detailed pipeline debugging."""
    global _PIPELINE_DEBUG, _DEBUG_MAX_SAMPLES
    _PIPELINE_DEBUG = True
    _DEBUG_MAX_SAMPLES = max_samples
    print(f"[PIPELINE DEBUG] Enabled - will log first {max_samples} samples in detail")

def disable_pipeline_debug():
    """Disable pipeline debugging."""
    global _PIPELINE_DEBUG
    _PIPELINE_DEBUG = False
    print("[PIPELINE DEBUG] Disabled")

def _should_log_sample() -> bool:
    """Check if we should log this sample."""
    global _DEBUG_SAMPLE_COUNT
    if not _PIPELINE_DEBUG:
        return False
    return _DEBUG_SAMPLE_COUNT < _DEBUG_MAX_SAMPLES

def _increment_sample_count():
    """Increment logged sample count."""
    global _DEBUG_SAMPLE_COUNT
    _DEBUG_SAMPLE_COUNT += 1

def _log_pipeline(stage: str, message: str, indent: int = 0):
    """Log a pipeline message."""
    if not _PIPELINE_DEBUG:
        return
    prefix = "  " * indent
    print(f"[PIPELINE {stage}] {prefix}{message}")


class XVRDataset:
    """
    Dataset class for loading XVR data in JSONL format.

    The XVR dataset contains multi-image vision tasks with questions and answers.
    Each sample includes multiple images and a text prompt.
    """

    def __init__(
        self,
        jsonl_path: str,
        image_base_dir: str,
        tokenizer,
        max_seq_length: int = 512,
        image_size: int = 224,
        max_samples: Optional[int] = None,
        shuffle_buffer_size: int = 1000,
    ):
        """
        Initialize XVR dataset loader.

        Args:
            jsonl_path: Path to JSONL file (train.jsonl, valid.jsonl, etc.)
            image_base_dir: Base directory containing image files
            tokenizer: SentencePiece tokenizer
            max_seq_length: Maximum sequence length for text
            image_size: Image size (height and width)
            max_samples: Maximum number of samples to use (None = all)
            shuffle_buffer_size: Buffer size for shuffling
        """
        self.jsonl_path = jsonl_path
        self.image_base_dir = Path(image_base_dir)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        self.max_samples = max_samples
        self.shuffle_buffer_size = shuffle_buffer_size

        # Load and validate dataset
        self._validate_paths()
        self.num_samples = self._count_samples()

    def _validate_paths(self):
        """Validate that paths exist."""
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")
        if not self.image_base_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_base_dir}")

    def _count_samples(self) -> int:
        """Count number of samples in JSONL file."""
        count = 0
        with open(self.jsonl_path, 'r') as f:
            for _ in f:
                count += 1
                if self.max_samples and count >= self.max_samples:
                    break
        return count

    def parse_sample(self, line: str) -> Dict:
        """
        Parse a single JSONL line into a structured sample.

        Args:
            line: JSON line from JSONL file

        Returns:
            Dictionary with parsed sample data
        """
        data = json.loads(line)

        # Extract images and text from prompt_blocks
        images = []
        text_parts = []

        for block in data['prompt_blocks']:
            if block['type'] == 'text':
                text_parts.append(block['text'])
            elif block['type'] == 'image_url':
                image_path = block['image_url']['url']
                images.append(image_path)

        # Combine text parts into a single prompt
        prompt = ''.join(text_parts)

        # Get ground truth answer
        answer = data.get('ground_truth_answer', '')

        result = {
            'sample_id': data['sample_id'],
            'task': data['task'],
            'images': images,
            'prompt': prompt,
            'answer': answer,
        }
        
        # Pipeline logging
        if _should_log_sample():
            _log_pipeline("1.PARSE", "=" * 60)
            _log_pipeline("1.PARSE", f"Sample ID: {result['sample_id']}")
            _log_pipeline("1.PARSE", f"Task: {result['task']}")
            _log_pipeline("1.PARSE", f"Number of images: {len(images)}")
            for i, img_path in enumerate(images):
                _log_pipeline("1.PARSE", f"  Image {i}: {img_path}", indent=1)
            _log_pipeline("1.PARSE", f"Prompt length: {len(prompt)} chars")
            _log_pipeline("1.PARSE", f"Prompt preview: {prompt[:100]}...")
            _log_pipeline("1.PARSE", f"Answer: '{answer}'")
        
        return result

    def load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from disk.

        Args:
            image_path: Relative path to image

        Returns:
            PIL Image
        """
        full_path = self.image_base_dir / image_path

        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")

        img = Image.open(full_path)
        
        # Pipeline logging
        if _should_log_sample():
            _log_pipeline("2.LOAD_IMG", f"Loaded: {image_path}")
            _log_pipeline("2.LOAD_IMG", f"  Original size: {img.size} (W x H)", indent=1)
            _log_pipeline("2.LOAD_IMG", f"  Mode: {img.mode}", indent=1)
        
        return img

    def get_tfdata(self, shuffle: bool = False, repeat: bool = False) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset for efficient data loading.

        Args:
            shuffle: Whether to shuffle the dataset
            repeat: Whether to repeat the dataset infinitely

        Returns:
            tf.data.Dataset
        """
        dataset = tf.data.TextLineDataset(self.jsonl_path)

        if self.max_samples:
            dataset = dataset.take(self.max_samples)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(self.shuffle_buffer_size, self.num_samples))

        if repeat:
            dataset = dataset.repeat()

        return dataset


def preprocess_single_image(image: Image.Image, size: int = 224) -> np.ndarray:
    """
    Preprocess a single image for PaliGemma model.

    Args:
        image: PIL Image
        size: Target size (height and width)

    Returns:
        Preprocessed image as numpy array in range [-1, 1]
    """
    # Convert to numpy array
    image = np.asarray(image)

    # Handle grayscale images
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)

    # Remove alpha channel if present
    image = image[..., :3]

    # Ensure RGB format
    assert image.shape[-1] == 3, f"Expected 3 channels, got {image.shape[-1]}"

    # Resize with bilinear interpolation
    image = tf.constant(image)
    image = tf.image.resize(
        image,
        (size, size),
        method='bilinear',
        antialias=True
    )

    # Normalize to [-1, 1]
    return image.numpy() / 127.5 - 1.0


def preprocess_image(
    image: Image.Image,
    size: int = 224,
) -> np.ndarray:
    """
    Preprocess image for PaliGemma model (backward compatible wrapper).

    Args:
        image: PIL Image
        size: Target size (height and width)

    Returns:
        Preprocessed image as numpy array in range [-1, 1]
    """
    return preprocess_single_image(image, size)


def preprocess_multi_images(
    images: list,
    size: int = 224,
    max_images: int = 6,
) -> Tuple[np.ndarray, int]:
    """
    Preprocess multiple images as separate frames for PaliGemma.

    IMPORTANT: big_vision PaliGemma handles multi-image by treating them as video frames.
    When image has shape [B, T, H, W, 3] (5D), it processes each frame separately
    and concatenates their tokens in the sequence.

    This is the CORRECT approach - NOT combining into a grid!

    Args:
        images: List of PIL Images
        size: Target output size (height and width)
        max_images: Maximum number of images to process

    Returns:
        Tuple of:
        - Preprocessed images as numpy array with shape [num_images, H, W, 3] in range [-1, 1]
        - Number of actual images (0 if DISABLE_IMAGES=1)
    """
    # Check if images are disabled via environment variable
    # Set DISABLE_IMAGES=1 to train without images (text-only training)
    disable_images = os.environ.get("DISABLE_IMAGES", "0") == "1"
    
    if disable_images:
        # Return empty image tensor with num_images=0
        # This will cause the model to skip image token generation
        # Shape: [0, H, W, 3] - empty array with correct dimensions
        empty_images = np.zeros((0, size, size, 3), dtype=np.float32)
        
        if _should_log_sample():
            _log_pipeline("3.PREPROCESS_IMGS", "DISABLE_IMAGES=1: Skipping image processing")
            _log_pipeline("3.PREPROCESS_IMGS", f"Returning empty images with shape: {empty_images.shape}")
            _log_pipeline("3.PREPROCESS_IMGS", f"num_images: 0 (images disabled)")
        
        return empty_images, 0
    
    if len(images) == 0:
        raise ValueError("At least one image is required")

    original_count = len(images)
    
    # Limit number of images
    images = images[:max_images]
    num_images = len(images)

    # Pipeline logging
    if _should_log_sample():
        _log_pipeline("3.PREPROCESS_IMGS", f"Input: {original_count} images, max_images={max_images}")
        _log_pipeline("3.PREPROCESS_IMGS", f"Using: {num_images} images (after limit)")

    # Preprocess each image individually
    processed_images = []
    for i, img in enumerate(images):
        processed = preprocess_single_image(img, size)
        processed_images.append(processed)
        
        if _should_log_sample():
            _log_pipeline("3.PREPROCESS_IMGS", f"  Image {i}: shape={processed.shape}, range=[{processed.min():.2f}, {processed.max():.2f}]", indent=1)

    # Stack into [num_images, H, W, 3]
    stacked = np.stack(processed_images, axis=0)

    if _should_log_sample():
        _log_pipeline("3.PREPROCESS_IMGS", f"Output stacked shape: {stacked.shape}")
        _log_pipeline("3.PREPROCESS_IMGS", f"  Expected: [{num_images}, {size}, {size}, 3]")

    return stacked, num_images


def preprocess_multi_images_grid(
    images: list,
    size: int = 224,
    grid_layout: str = "auto",
) -> np.ndarray:
    """
    DEPRECATED: This combines images into a grid, which loses information.
    Use preprocess_multi_images() instead for proper multi-image handling.

    Preprocess multiple images by combining them into a single grid image.
    Kept for backward compatibility only.
    """
    import warnings
    warnings.warn(
        "preprocess_multi_images_grid() combines images into a grid which loses information. "
        "Use preprocess_multi_images() for proper multi-image handling.",
        DeprecationWarning
    )

    if len(images) == 0:
        raise ValueError("At least one image is required")

    if len(images) == 1:
        return preprocess_single_image(images[0], size)

    # Preprocess each image individually first (to handle grayscale, alpha, etc.)
    processed_images = []
    for img in images:
        img_array = np.asarray(img)

        # Handle grayscale
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)

        # Remove alpha channel
        img_array = img_array[..., :3]

        processed_images.append(img_array)

    num_images = len(processed_images)

    # Determine grid layout
    if grid_layout == "auto":
        if num_images == 2:
            grid_layout = "horizontal"  # Side by side
        elif num_images <= 4:
            grid_layout = "2x2"
        else:
            # For more than 4 images, use 2-row layout
            grid_layout = "2xN"

    # Resize images to same size for grid
    # Use a larger intermediate size for better quality
    if grid_layout == "horizontal":
        # Side by side: each image gets half width
        cell_h, cell_w = size, size // num_images
    elif grid_layout == "vertical":
        cell_h, cell_w = size // num_images, size
    elif grid_layout == "2x2":
        cell_h, cell_w = size // 2, size // 2
    elif grid_layout == "2xN":
        cols = (num_images + 1) // 2
        cell_h, cell_w = size // 2, size // cols
    else:
        cell_h, cell_w = size // 2, size // 2

    # Resize each image to cell size
    resized_images = []
    for img_array in processed_images:
        img_tensor = tf.constant(img_array, dtype=tf.float32)
        resized = tf.image.resize(
            img_tensor,
            (cell_h, cell_w),
            method='bilinear',
            antialias=True
        )
        resized_images.append(resized.numpy())

    # Combine images into grid
    if grid_layout == "horizontal":
        combined = np.concatenate(resized_images, axis=1)
    elif grid_layout == "vertical":
        combined = np.concatenate(resized_images, axis=0)
    elif grid_layout == "2x2":
        # Pad if necessary for 2x2
        while len(resized_images) < 4:
            # Add black padding image
            resized_images.append(np.zeros_like(resized_images[0]))
        row1 = np.concatenate(resized_images[0:2], axis=1)
        row2 = np.concatenate(resized_images[2:4], axis=1)
        combined = np.concatenate([row1, row2], axis=0)
    elif grid_layout == "2xN":
        cols = (num_images + 1) // 2
        # Pad if necessary
        while len(resized_images) < cols * 2:
            resized_images.append(np.zeros_like(resized_images[0]))
        row1 = np.concatenate(resized_images[0:cols], axis=1)
        row2 = np.concatenate(resized_images[cols:cols*2], axis=1)
        combined = np.concatenate([row1, row2], axis=0)
    else:
        combined = resized_images[0]

    # Final resize to ensure exact target dimensions
    combined_tensor = tf.constant(combined, dtype=tf.float32)
    final = tf.image.resize(
        combined_tensor,
        (size, size),
        method='bilinear',
        antialias=True
    )

    # Normalize to [-1, 1]
    return final.numpy() / 127.5 - 1.0


def get_image_token_count(image_size: int = 224, patch_size: int = 14) -> int:
    """
    Calculate number of tokens per image for PaliGemma ViT encoder.

    PaliGemma uses a ViT encoder that splits the image into patches.
    Each patch becomes one token.

    Args:
        image_size: Image size (e.g., 224)
        patch_size: Patch size (e.g., 14 for So400m/14)

    Returns:
        Number of tokens per image
    """
    num_patches_per_side = image_size // patch_size
    return num_patches_per_side * num_patches_per_side  # 224/14 = 16 -> 16*16 = 256


def preprocess_tokens(
    prefix: str,
    suffix: Optional[str] = None,
    seqlen: Optional[int] = None,
    tokenizer=None,
    num_images: int = 1,
    image_size: int = 224,
    patch_size: int = 14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tokenize and prepare text for PaliGemma model.

    PaliGemma uses a prefix (full attention) and suffix (causal attention) structure.

    IMPORTANT: The model prepends image tokens to the text sequence internally.
    We need to account for this in mask_ar and mask_loss so that:
    - Image tokens use full attention (mask_ar=0) and are not trained (mask_loss=0)
    - Prefix text uses full attention (mask_ar=0) and is not trained (mask_loss=0)
    - Suffix text uses causal attention (mask_ar=1) and IS trained (mask_loss=1)

    Args:
        prefix: Prefix text (empty by default, can include instructions like "answer en")
        suffix: Suffix text (e.g., the answer)
        seqlen: Maximum sequence length for TEXT only (will pad if needed)
        tokenizer: SentencePiece tokenizer
        num_images: Number of images (each adds tokens to the sequence)
        image_size: Image size for calculating token count
        patch_size: Patch size (14 for So400m/14)

    Returns:
        Tuple of (tokens, mask_ar, mask_loss, mask_input)
        Note: These are for TEXT tokens only. The model handles image tokens internally.
    """
    separator = "\n"

    # Tokenize prefix with BOS token
    tokens = tokenizer.encode(prefix, add_bos=True) + tokenizer.encode(separator)
    mask_ar = [0] * len(tokens)
    mask_loss = [0] * len(tokens)

    # Calculate image token count
    image_tokens_per_image = get_image_token_count(image_size, patch_size)
    total_image_tokens = image_tokens_per_image * num_images

    # Pipeline logging
    if _should_log_sample():
        _log_pipeline("4.TOKENIZE", f"Tokenizing text...")
        _log_pipeline("4.TOKENIZE", f"  num_images: {num_images}")
        _log_pipeline("4.TOKENIZE", f"  image_tokens_per_image: {image_tokens_per_image} ({image_size}x{image_size} / {patch_size}x{patch_size})")
        _log_pipeline("4.TOKENIZE", f"  total_image_tokens: {total_image_tokens}")
        _log_pipeline("4.TOKENIZE", f"  prefix: '{prefix[:80]}...'")
        _log_pipeline("4.TOKENIZE", f"  prefix tokens (with BOS + separator): {len(tokens)}")

    # Add suffix if provided
    suffix_token_count = 0
    if suffix:
        suffix_tokens = tokenizer.encode(suffix, add_eos=True)
        suffix_token_count = len(suffix_tokens)
        tokens += suffix_tokens
        mask_ar += [1] * len(suffix_tokens)
        mask_loss += [1] * len(suffix_tokens)

        if _should_log_sample():
            _log_pipeline("4.TOKENIZE", f"  suffix (answer): '{suffix}'")
            _log_pipeline("4.TOKENIZE", f"  suffix tokens (with EOS): {suffix_token_count}")

    # Mark input positions
    mask_input = [1] * len(tokens)

    # Track original length for truncation warning
    original_len = len(tokens)

    # Pad to sequence length if specified
    truncated = False
    if seqlen:
        if len(tokens) > seqlen:
            truncated = True

        padding = [0] * max(0, seqlen - len(tokens))
        tokens = tokens[:seqlen] + padding
        mask_ar = mask_ar[:seqlen] + padding
        mask_loss = mask_loss[:seqlen] + padding
        mask_input = mask_input[:seqlen] + padding

    if _should_log_sample():
        _log_pipeline("4.TOKENIZE", f"  --- Final text token breakdown ---")
        _log_pipeline("4.TOKENIZE", f"  Total text tokens (before padding): {original_len}")
        _log_pipeline("4.TOKENIZE", f"  Truncated: {truncated}")
        _log_pipeline("4.TOKENIZE", f"  Padded to seqlen: {seqlen}")
        _log_pipeline("4.TOKENIZE", f"  mask_ar nonzero (causal tokens): {sum(mask_ar)}")
        _log_pipeline("4.TOKENIZE", f"  mask_loss nonzero (trainable tokens): {sum(mask_loss)}")
        _log_pipeline("4.TOKENIZE", f"  --- Total sequence (image + text) ---")
        _log_pipeline("4.TOKENIZE", f"  Image tokens: {total_image_tokens}")
        _log_pipeline("4.TOKENIZE", f"  Text tokens: {original_len}")
        _log_pipeline("4.TOKENIZE", f"  TOTAL: {total_image_tokens + original_len}")

    return (
        np.array(tokens, dtype=np.int32),
        np.array(mask_ar, dtype=np.int32),
        np.array(mask_loss, dtype=np.int32),
        np.array(mask_input, dtype=np.int32),
    )


def _validate_batch_images(
    batch_images: np.ndarray,
    samples: list,
    batch_num_images: np.ndarray,
    max_images: int,
) -> None:
    """
    Validate that images in batch match the original samples.
    
    This helps catch data corruption or mixing issues early.
    
    Args:
        batch_images: Batched images [B, max_images, H, W, 3]
        samples: Original sample list
        batch_num_images: Number of images per sample [B]
        max_images: Maximum number of images
    """
    import os
    # Only validate occasionally to avoid performance impact
    validate = os.environ.get("VALIDATE_BATCH_IMAGES", "0") == "1"
    if not validate:
        return
    
    batch_size = len(samples)
    mismatches = []
    
    for i in range(batch_size):
        num_images = int(batch_num_images[i])
        sample = samples[i]
        sample_id = sample.get('sample_id', f'sample_{i}')
        original_images = sample.get('image', None)
        
        if original_images is None or num_images == 0:
            continue
        
        # Compare each image
        for img_idx in range(num_images):
            batch_img = batch_images[i, img_idx]
            original_img = original_images[img_idx]
            
            # Check if images match (allowing for small floating point differences)
            if not np.allclose(batch_img, original_img, atol=1e-5, rtol=1e-5):
                # Check if it's just padding (all zeros)
                if np.allclose(batch_img, 0.0, atol=1e-5):
                    mismatches.append(
                        f"Sample {i} ({sample_id}), image {img_idx}: "
                        f"Batch has zeros (padding) but original has data"
                    )
                elif np.allclose(original_img, 0.0, atol=1e-5):
                    mismatches.append(
                        f"Sample {i} ({sample_id}), image {img_idx}: "
                        f"Original has zeros but batch has data"
                    )
                else:
                    # Actual mismatch
                    max_diff = np.max(np.abs(batch_img - original_img))
                    mismatches.append(
                        f"Sample {i} ({sample_id}), image {img_idx}: "
                        f"Images don't match! Max diff: {max_diff:.6f}"
                    )
        
        # Check that padding slots are zeros
        for img_idx in range(num_images, max_images):
            batch_img = batch_images[i, img_idx]
            if not np.allclose(batch_img, 0.0, atol=1e-5):
                mismatches.append(
                    f"Sample {i} ({sample_id}), image {img_idx}: "
                    f"Padding slot is not zero!"
                )
    
    if mismatches:
        print(f"\n[VALIDATION ERROR] Found {len(mismatches)} image mismatches in batch:")
        for mismatch in mismatches[:10]:  # Show first 10
            print(f"  {mismatch}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more mismatches")
        print("[VALIDATION ERROR] This indicates data corruption or mixing issues!\n")
    else:
        # Only print success message occasionally
        import random
        if random.random() < 0.50:  # 50% chance
            print(f"[VALIDATION] Batch image validation passed for {batch_size} samples")


def collate_batch(samples: list, max_images: int = 6, image_size: int = 224) -> Dict[str, np.ndarray]:
    """
    Collate a list of samples into a batch for training.

    This handles the tricky part of batching multi-image samples where
    each sample can have a different number of images.

    For big_vision PaliGemma, images need to be in shape [B, T, H, W, 3]
    where T is the number of frames/images. We pad to max_images.

    Args:
        samples: List of sample dictionaries from the iterator
        max_images: Maximum number of images to pad to
        image_size: Image size

    Returns:
        Batched dictionary with:
        - image: [B, max_images, H, W, 3]
        - text: [B, seq_len]
        - mask_ar: [B, seq_len]
        - mask_loss: [B, seq_len]
        - num_images: [B] actual number of images per sample
    """
    batch_size = len(samples)

    # Get dimensions from first sample
    seq_len = samples[0]['text'].shape[0]

    # Pipeline logging - only log first few batches
    _should_log_collate = _PIPELINE_DEBUG and _DEBUG_SAMPLE_COUNT <= _DEBUG_MAX_SAMPLES + 10
    if _should_log_collate:
        _log_pipeline("6.COLLATE", "=" * 60)
        _log_pipeline("6.COLLATE", f"Collating {batch_size} samples into batch")
        _log_pipeline("6.COLLATE", f"  max_images: {max_images}")
        _log_pipeline("6.COLLATE", f"  image_size: {image_size}")
        _log_pipeline("6.COLLATE", f"  seq_len: {seq_len}")

    # Initialize batch arrays
    batch_images = np.zeros((batch_size, max_images, image_size, image_size, 3), dtype=np.float32)
    batch_text = np.zeros((batch_size, seq_len), dtype=np.int32)
    batch_mask_ar = np.zeros((batch_size, seq_len), dtype=np.int32)
    batch_mask_loss = np.zeros((batch_size, seq_len), dtype=np.int32)
    batch_num_images = np.zeros((batch_size,), dtype=np.int32)

    for i, sample in enumerate(samples):
        num_images = sample['num_images']
        batch_num_images[i] = num_images

        # Copy images (sample['image'] has shape [num_images, H, W, 3])
        # If num_images=0 (DISABLE_IMAGES=1), sample['image'] is empty [0, H, W, 3]
        # and batch_images[i, :0] is an empty slice, so this is safe
        if num_images > 0:
            batch_images[i, :num_images] = sample['image']

        # Copy text and masks
        batch_text[i] = sample['text']
        batch_mask_ar[i] = sample['mask_ar']
        batch_mask_loss[i] = sample['mask_loss']

        # Pipeline logging per sample
        if _should_log_collate:
            sample_id = sample.get('sample_id', f'sample_{i}')
            _log_pipeline("6.COLLATE", f"  Sample {i} ({sample_id}):")
            _log_pipeline("6.COLLATE", f"    num_images: {num_images}")
            _log_pipeline("6.COLLATE", f"    input image shape: {sample['image'].shape}")
            _log_pipeline("6.COLLATE", f"    text nonzero: {np.count_nonzero(sample['text'])}")
            _log_pipeline("6.COLLATE", f"    mask_loss nonzero: {np.count_nonzero(sample['mask_loss'])}")
            _log_pipeline("6.COLLATE", f"    ground_truth: '{sample.get('ground_truth', 'N/A')}'")

    if _should_log_collate:
        _log_pipeline("6.COLLATE", f"  --- Final Batch Shape ---")
        _log_pipeline("6.COLLATE", f"  batch_images: {batch_images.shape} (expected: [{batch_size}, {max_images}, {image_size}, {image_size}, 3])")
        _log_pipeline("6.COLLATE", f"  batch_text: {batch_text.shape}")
        _log_pipeline("6.COLLATE", f"  batch_mask_ar: {batch_mask_ar.shape}")
        _log_pipeline("6.COLLATE", f"  batch_mask_loss: {batch_mask_loss.shape}")
        _log_pipeline("6.COLLATE", f"  batch_num_images: {batch_num_images}")
        _log_pipeline("6.COLLATE", "=" * 60)

    # Validation: Verify that images in batch match original samples
    # This helps catch any data corruption or mixing issues
    _validate_batch_images(batch_images, samples, batch_num_images, max_images)

    return {
        'image': batch_images,
        'text': batch_text,
        'mask_ar': batch_mask_ar,
        'mask_loss': batch_mask_loss,
        'num_images': batch_num_images,
    }


def collate_eval_batch(samples: list, max_images: int = 6, image_size: int = 224) -> Dict[str, np.ndarray]:
    """
    Collate a list of samples into a batch for evaluation.

    Similar to collate_batch but preserves evaluation metadata.
    """
    batch_size = len(samples)
    seq_len = samples[0]['text'].shape[0]

    batch_images = np.zeros((batch_size, max_images, image_size, image_size, 3), dtype=np.float32)
    batch_text = np.zeros((batch_size, seq_len), dtype=np.int32)
    batch_mask_ar = np.zeros((batch_size, seq_len), dtype=np.int32)
    batch_mask_input = np.zeros((batch_size, seq_len), dtype=np.int32)
    batch_num_images = np.zeros((batch_size,), dtype=np.int32)

    sample_ids = []
    ground_truths = []
    input_prompts = []

    for i, sample in enumerate(samples):
        num_images = sample['num_images']
        batch_num_images[i] = num_images
        batch_images[i, :num_images] = sample['image']
        batch_text[i] = sample['text']
        batch_mask_ar[i] = sample['mask_ar']
        batch_mask_input[i] = sample['mask_input']

        sample_ids.append(sample['sample_id'])
        ground_truths.append(sample['ground_truth'])
        input_prompts.append(sample['input_prompt'])

    return {
        'image': batch_images,
        'text': batch_text,
        'mask_ar': batch_mask_ar,
        'mask_input': batch_mask_input,
        'num_images': batch_num_images,
        'sample_ids': sample_ids,
        'ground_truths': ground_truths,
        'input_prompts': input_prompts,
    }


def postprocess_tokens(tokens: np.ndarray, tokenizer) -> str:
    """
    Detokenize model output back to text.

    Args:
        tokens: Token IDs from model
        tokenizer: SentencePiece tokenizer

    Returns:
        Decoded text string
    """
    tokens = tokens.tolist() if isinstance(tokens, np.ndarray) else tokens

    # Remove tokens at and after EOS
    try:
        eos_pos = tokens.index(tokenizer.eos_id())
        tokens = tokens[:eos_pos]
    except ValueError:
        pass

    return tokenizer.decode(tokens)


def create_train_iterator(
    dataset: XVRDataset,
    batch_size: int,
    prompt_prefix: str = "",  # Empty by default (matching Google's official tutorial)
    max_images: int = 6,
) -> Iterator[Dict[str, np.ndarray]]:
    """
    Create a training data iterator.

    Args:
        dataset: XVRDataset instance
        batch_size: Number of examples per batch
        prompt_prefix: Prefix for all prompts (empty by default, can be set to "answer en" etc.)
        max_images: Maximum number of images per sample

    Yields:
        Dictionary with sample data (NOT batched - batching happens in training loop)
    """
    tf_dataset = dataset.get_tfdata(shuffle=True, repeat=True)

    for line in tf_dataset.as_numpy_iterator():
        sample = dataset.parse_sample(line.decode('utf-8'))

        # Check if images are disabled via environment variable
        disable_images = os.environ.get("DISABLE_IMAGES", "0") == "1"
        
        if disable_images:
            # Skip image loading and create empty images
            # preprocess_multi_images will handle the empty list and return num_images=0
            images, num_images = preprocess_multi_images(
                [],  # Empty list - will be handled by preprocess_multi_images
                size=dataset.image_size,
                max_images=max_images,
            )
        else:
            if not sample['images']:
                continue

            # Load and process ALL images as separate frames (NOT grid!)
            try:
                loaded_images = []
                for img_path in sample['images']:
                    img = dataset.load_image(img_path)
                    loaded_images.append(img)

                # Process images as separate frames [num_images, H, W, 3]
                # This is the CORRECT approach for multi-image!
                images, num_images = preprocess_multi_images(
                    loaded_images,
                    size=dataset.image_size,
                    max_images=max_images,
                )
            except Exception as e:
                print(f"Warning: Failed to load images for sample {sample.get('sample_id', 'unknown')}: {e}")
                continue

        # Combine prompt_prefix (if any) with the actual question
        # Format: "{prompt_prefix}{actual_question}" (with space if prefix exists)
        if prompt_prefix:
            full_prefix = f"{prompt_prefix} {sample['prompt']}"
        else:
            full_prefix = sample['prompt']

        suffix = sample['answer']
        tokens, mask_ar, mask_loss, _ = preprocess_tokens(
            prefix=full_prefix,
            suffix=suffix,
            seqlen=dataset.max_seq_length,
            tokenizer=dataset.tokenizer,
            num_images=num_images,
            image_size=dataset.image_size,
        )

        # Pipeline logging - Final yield
        if _should_log_sample():
            image_tokens_per_image = get_image_token_count(dataset.image_size, patch_size=14)
            total_image_tokens = image_tokens_per_image * num_images
            _log_pipeline("5.YIELD", "=" * 60)
            _log_pipeline("5.YIELD", f"Final sample ready to yield:")
            _log_pipeline("5.YIELD", f"  sample_id: {sample.get('sample_id', None)}")
            _log_pipeline("5.YIELD", f"  --- Images ---")
            _log_pipeline("5.YIELD", f"  image.shape: {images.shape} (expected: [{num_images}, 224, 224, 3])")
            _log_pipeline("5.YIELD", f"  num_images: {num_images}")
            _log_pipeline("5.YIELD", f"  image_tokens_per_image: {image_tokens_per_image}")
            _log_pipeline("5.YIELD", f"  total_image_tokens: {total_image_tokens}")
            _log_pipeline("5.YIELD", f"  --- Text ---")
            _log_pipeline("5.YIELD", f"  text.shape: {np.asarray(tokens).shape}")
            _log_pipeline("5.YIELD", f"  text nonzero tokens: {np.count_nonzero(tokens)}")
            _log_pipeline("5.YIELD", f"  mask_ar nonzero (causal): {np.count_nonzero(mask_ar)}")
            _log_pipeline("5.YIELD", f"  mask_loss nonzero (trainable): {np.count_nonzero(mask_loss)}")
            _log_pipeline("5.YIELD", f"  --- Metadata ---")
            _log_pipeline("5.YIELD", f"  input_prompt: '{full_prefix[:60]}...'")
            _log_pipeline("5.YIELD", f"  ground_truth: '{suffix}'")
            _log_pipeline("5.YIELD", f"  --- Total Sequence ---")
            _log_pipeline("5.YIELD", f"  TOTAL tokens (image + text): {total_image_tokens + np.count_nonzero(tokens)}")
            _log_pipeline("5.YIELD", "=" * 60)
            _increment_sample_count()

        yield {
            'image': images,  # Shape: [num_images, H, W, 3]
            'text': np.asarray(tokens),
            'mask_ar': np.asarray(mask_ar),
            'mask_loss': np.asarray(mask_loss),
            'num_images': num_images,
            # Add metadata for debugging
            'input_prompt': full_prefix,  # Original input text
            'ground_truth': suffix,  # Original answer
            'sample_id': sample.get('sample_id', None),
        }


def create_eval_iterator(
    dataset: XVRDataset,
    batch_size: int,
    prompt_prefix: str = "",  # Empty by default (matching Google's official tutorial)
    num_examples: Optional[int] = None,
    max_images: int = 6,
) -> Iterator[Dict[str, np.ndarray]]:
    """
    Create an evaluation data iterator.

    Args:
        dataset: XVRDataset instance
        batch_size: Number of examples per batch
        prompt_prefix: Prefix for all prompts (empty by default, can be set to "answer en" etc.)
        num_examples: Maximum number of examples to yield (None = all)
        max_images: Maximum number of images per sample

    Yields:
        Dictionary with sample data for evaluation
    """
    tf_dataset = dataset.get_tfdata(shuffle=False, repeat=False)

    count = 0
    for line in tf_dataset.as_numpy_iterator():
        if num_examples and count >= num_examples:
            break

        sample = dataset.parse_sample(line.decode('utf-8'))

        # Check if images are disabled via environment variable
        disable_images = os.environ.get("DISABLE_IMAGES", "0") == "1"
        
        if disable_images:
            # Skip image loading and create empty images
            # preprocess_multi_images will handle the empty list and return num_images=0
            images, num_images = preprocess_multi_images(
                [],  # Empty list - will be handled by preprocess_multi_images
                size=dataset.image_size,
                max_images=max_images,
            )
        else:
            if not sample['images']:
                continue

            # Load and process ALL images as separate frames (NOT grid!)
            try:
                loaded_images = []
                for img_path in sample['images']:
                    img = dataset.load_image(img_path)
                    loaded_images.append(img)

                # Process images as separate frames [num_images, H, W, 3]
                images, num_images = preprocess_multi_images(
                    loaded_images,
                    size=dataset.image_size,
                    max_images=max_images,
                )
            except Exception as e:
                print(f"Warning: Failed to load images for sample {sample.get('sample_id', 'unknown')}: {e}")
                continue

        # Combine prompt_prefix (if any) with the actual question
        # Format: "{prompt_prefix}{actual_question}" (with space if prefix exists)
        if prompt_prefix:
            full_prefix = f"{prompt_prefix} {sample['prompt']}"
        else:
            full_prefix = sample['prompt']

        tokens, mask_ar, _, mask_input = preprocess_tokens(
            prefix=full_prefix,
            suffix=None,
            seqlen=dataset.max_seq_length,
            tokenizer=dataset.tokenizer,
            num_images=num_images,
            image_size=dataset.image_size,
        )

        # DEBUG: Check token counts for evaluation
        if random.random() < 0.01:
            image_tokens_per_image = get_image_token_count(dataset.image_size, patch_size=14)
            total_image_tokens = image_tokens_per_image * num_images
            print(f"\n[DEBUG create_eval_iterator] After preprocess_tokens:")
            print(f"  num_images: {num_images}")
            print(f"  image shape: {images.shape}")
            print(f"  image_tokens_per_image: {image_tokens_per_image}")
            print(f"  total_image_tokens: {total_image_tokens}")
            print(f"  text tokens shape: {np.asarray(tokens).shape}")
            print(f"  text tokens nonzero: {np.count_nonzero(tokens)}")
            print(f"  total sequence length (image + text): {total_image_tokens + np.count_nonzero(tokens)}")
            print(f"  sample_id: {sample['sample_id']}")
            print(f"  prompt length: {len(full_prefix)} chars")

        yield {
            'image': images,  # Shape: [num_images, H, W, 3]
            'text': np.asarray(tokens),
            'mask_ar': np.asarray(mask_ar),
            'mask_input': np.asarray(mask_input),
            'sample_id': sample['sample_id'],
            'ground_truth': sample['answer'],
            'input_prompt': full_prefix,  # For debugging: the actual prompt sent to model
            'num_images': num_images,  # Number of images
        }

        count += 1


class PrefetchIterator:
    """
    Wraps an iterator to prefetch items in a background thread.
    This allows data loading to happen in parallel with GPU computation.
    """

    def __init__(self, iterator: Iterator, prefetch_size: int = 2):
        """
        Args:
            iterator: Source iterator to prefetch from
            prefetch_size: Number of items to prefetch (default: 2)
        """
        self.iterator = iterator
        self.prefetch_size = prefetch_size
        self.queue = queue.Queue(maxsize=prefetch_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()

    def _prefetch_worker(self):
        """Background thread that prefetches items into the queue."""
        try:
            for item in self.iterator:
                if self.stop_event.is_set():
                    break
                self.queue.put(item)
            # Signal end of iterator
            self.queue.put(None)
        except Exception as e:
            self.queue.put(e)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def stop(self):
        """Stop the prefetch thread."""
        self.stop_event.set()


def prefetch_iterator(iterator: Iterator, prefetch_size: int = 2) -> Iterator:
    """
    Wrap an iterator to prefetch items in background.

    Usage:
        train_iter = prefetch_iterator(create_train_iterator(...), prefetch_size=2)
        for batch in train_iter:
            ...

    Args:
        iterator: Source iterator
        prefetch_size: Number of items to prefetch

    Returns:
        Iterator that prefetches in background
    """
    return PrefetchIterator(iterator, prefetch_size)

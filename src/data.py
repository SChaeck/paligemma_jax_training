"""
Data loading and preprocessing utilities for XVR dataset.

This module handles loading the XVR dataset in JSONL format and preprocessing
images and text for PaliGemma training.
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import Config


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

        return {
            'sample_id': data['sample_id'],
            'task': data['task'],
            'images': images,
            'prompt': prompt,
            'answer': answer,
        }

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

        return Image.open(full_path)

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
        - Number of actual images
    """
    if len(images) == 0:
        raise ValueError("At least one image is required")

    # Limit number of images
    images = images[:max_images]
    num_images = len(images)

    # Preprocess each image individually
    processed_images = []
    for img in images:
        processed = preprocess_single_image(img, size)
        processed_images.append(processed)

    # Stack into [num_images, H, W, 3]
    stacked = np.stack(processed_images, axis=0)

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

    # Add suffix if provided
    if suffix:
        suffix_tokens = tokenizer.encode(suffix, add_eos=True)
        tokens += suffix_tokens
        mask_ar += [1] * len(suffix_tokens)
        mask_loss += [1] * len(suffix_tokens)

    # Mark input positions
    mask_input = [1] * len(tokens)

    # Track original length for truncation warning
    original_len = len(tokens)

    # Pad to sequence length if specified
    if seqlen:
        # Warn if truncation is happening (important for debugging)
        if len(tokens) > seqlen:
            # Only warn occasionally to avoid spam
            import random
            if random.random() < 0.01:  # 1% of truncations
                print(f"Warning: Truncating sequence from {len(tokens)} to {seqlen} tokens. "
                      f"Consider increasing MAX_SEQ_LENGTH.")

        padding = [0] * max(0, seqlen - len(tokens))
        tokens = tokens[:seqlen] + padding
        mask_ar = mask_ar[:seqlen] + padding
        mask_loss = mask_loss[:seqlen] + padding
        mask_input = mask_input[:seqlen] + padding

    return (
        np.array(tokens, dtype=np.int32),
        np.array(mask_ar, dtype=np.int32),
        np.array(mask_loss, dtype=np.int32),
        np.array(mask_input, dtype=np.int32),
    )


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
        batch_images[i, :num_images] = sample['image']

        # Copy text and masks
        batch_text[i] = sample['text']
        batch_mask_ar[i] = sample['mask_ar']
        batch_mask_loss[i] = sample['mask_loss']

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

        suffix = sample['answer'].lower()
        tokens, mask_ar, mask_loss, _ = preprocess_tokens(
            prefix=full_prefix,
            suffix=suffix,
            seqlen=dataset.max_seq_length,
            tokenizer=dataset.tokenizer,
            num_images=num_images,
            image_size=dataset.image_size,
        )

        yield {
            'image': images,  # Shape: [num_images, H, W, 3]
            'text': np.asarray(tokens),
            'mask_ar': np.asarray(mask_ar),
            'mask_loss': np.asarray(mask_loss),
            'num_images': num_images,
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

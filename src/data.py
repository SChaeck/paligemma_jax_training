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
        max_seq_length: int = 256,
        image_size: int = 224,
        max_samples: Optional[int] = None,
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
        """
        self.jsonl_path = jsonl_path
        self.image_base_dir = Path(image_base_dir)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        self.max_samples = max_samples

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
            dataset = dataset.shuffle(buffer_size=min(1000, self.num_samples))

        if repeat:
            dataset = dataset.repeat()

        return dataset


def preprocess_image(image: Image.Image, size: int = 224) -> np.ndarray:
    """
    Preprocess image for PaliGemma model.

    PaliGemma expects images resized to size x size in range [-1, 1].

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


def preprocess_tokens(
    prefix: str,
    suffix: Optional[str] = None,
    seqlen: Optional[int] = None,
    tokenizer=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tokenize and prepare text for PaliGemma model.

    PaliGemma uses a prefix (full attention) and suffix (causal attention) structure.

    Args:
        prefix: Prefix text (e.g., "answer en")
        suffix: Suffix text (e.g., the answer)
        seqlen: Maximum sequence length (will pad if needed)
        tokenizer: SentencePiece tokenizer

    Returns:
        Tuple of (tokens, mask_ar, mask_loss, mask_input)
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

    # Pad to sequence length if specified
    if seqlen:
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
    prompt_prefix: str = "answer en",
) -> Iterator[Dict[str, np.ndarray]]:
    """
    Create a training data iterator.

    Args:
        dataset: XVRDataset instance
        batch_size: Number of examples per batch
        prompt_prefix: Prefix for all prompts

    Yields:
        Dictionary with batched data
    """
    tf_dataset = dataset.get_tfdata(shuffle=True, repeat=True)

    for line in tf_dataset.as_numpy_iterator():
        sample = dataset.parse_sample(line.decode('utf-8'))

        if not sample['images']:
            continue

        try:
            image = dataset.load_image(sample['images'][0])
            image = preprocess_image(image, size=dataset.image_size)
        except Exception as e:
            print(f"Warning: Failed to load image {sample['images'][0]}: {e}")
            continue

        suffix = sample['answer'].lower()
        tokens, mask_ar, mask_loss, _ = preprocess_tokens(
            prefix=prompt_prefix,
            suffix=suffix,
            seqlen=dataset.max_seq_length,
            tokenizer=dataset.tokenizer,
        )

        yield {
            'image': np.asarray(image),
            'text': np.asarray(tokens),
            'mask_ar': np.asarray(mask_ar),
            'mask_loss': np.asarray(mask_loss),
        }


def create_eval_iterator(
    dataset: XVRDataset,
    batch_size: int,
    prompt_prefix: str = "answer en",
    num_examples: Optional[int] = None,
) -> Iterator[Dict[str, np.ndarray]]:
    """
    Create an evaluation data iterator.

    Args:
        dataset: XVRDataset instance
        batch_size: Number of examples per batch
        prompt_prefix: Prefix for all prompts
        num_examples: Maximum number of examples to yield (None = all)

    Yields:
        Dictionary with batched data for evaluation
    """
    tf_dataset = dataset.get_tfdata(shuffle=False, repeat=False)

    count = 0
    for line in tf_dataset.as_numpy_iterator():
        if num_examples and count >= num_examples:
            break

        sample = dataset.parse_sample(line.decode('utf-8'))

        if not sample['images']:
            continue

        try:
            image = dataset.load_image(sample['images'][0])
            image = preprocess_image(image, size=dataset.image_size)
        except Exception as e:
            print(f"Warning: Failed to load image {sample['images'][0]}: {e}")
            continue

        tokens, mask_ar, _, mask_input = preprocess_tokens(
            prefix=prompt_prefix,
            suffix=None,
            seqlen=dataset.max_seq_length,
            tokenizer=dataset.tokenizer,
        )

        yield {
            'image': np.asarray(image),
            'text': np.asarray(tokens),
            'mask_ar': np.asarray(mask_ar),
            'mask_input': np.asarray(mask_input),
            'sample_id': sample['sample_id'],
            'ground_truth': sample['answer'],
        }

        count += 1

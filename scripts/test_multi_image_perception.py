"""
Test script to verify if the model can perceive and describe multiple images.

This script loads a trained checkpoint and tests if it can:
1. See each image individually
2. Describe what's in each image
3. Differentiate between multiple images
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import numpy as np
from PIL import Image

from src.config import load_config
from src.model import load_paligemma_model
from src.data import XVRDataset, collate_eval_batch, preprocess_tokens, postprocess_tokens


def load_checkpoint(checkpoint_path, model, params, config):
    """
    Load checkpoint weights into params structure.
    
    This function properly handles the checkpoint format where parameters
    are saved with 'params/' prefix and need to be merged into the existing
    params structure.
    """
    from src.model import setup_big_vision
    setup_big_vision(config)
    import big_vision.utils
    import jax.tree_util as tree_util

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load the checkpoint
    loaded = np.load(checkpoint_path)

    # Get flattened current params with paths
    flat_params_list = big_vision.utils.tree_flatten_with_names(params)[0]

    # Build mapping from checkpoint keys to values
    checkpoint_params = {}
    for key in loaded.files:
        # Remove 'params/' prefix if present (checkpoints are saved with this prefix)
        param_key = key.replace('params/', '')
        checkpoint_params[param_key] = loaded[key]

    # Map checkpoint values back into tree structure
    leaves, treedef = tree_util.tree_flatten(params)
    
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

    print(f"Checkpoint loaded successfully")
    return updated_params


def prepare_inference_batch(prompt, tokenizer, num_images, image_batch, image_size=224, max_seq_len=512):
    """
    Prepare a batch for inference using the SAME tokenization as training.

    This ensures that the test uses the exact same data preprocessing pipeline
    as training, including proper mask_ar and mask_input setup.

    Args:
        prompt: Text prompt (prefix, no suffix for inference)
        tokenizer: SentencePiece tokenizer
        num_images: Number of images in the batch
        image_batch: Image tensor [1, max_images, H, W, 3]
        image_size: Image size for token calculation
        max_seq_len: Maximum sequence length

    Returns:
        Dictionary with properly formatted batch
    """
    # Use the SAME preprocess_tokens function as training
    # For inference: prefix=prompt, suffix=None (no answer to generate)
    tokens, mask_ar, mask_loss, mask_input = preprocess_tokens(
        prefix=prompt,
        suffix=None,  # No suffix for inference
        seqlen=max_seq_len,
        tokenizer=tokenizer,
        num_images=num_images,
        image_size=image_size,
        patch_size=14,
    )

    # Create batch (batch_size=1)
    batch = {
        'image': image_batch,  # [1, max_images, H, W, 3]
        'text': tokens[None, ...],  # [1, seq_len]
        'mask_ar': mask_ar[None, ...],  # [1, seq_len]
        'mask_input': mask_input[None, ...],  # [1, seq_len]
        '_mask': np.ones(1, dtype=bool),
        'num_images': np.array([num_images], dtype=np.int32),
    }

    return batch, len(tokenizer.encode(prompt, add_bos=True))


def test_multi_image_perception(
    model, params, decode_fn, tokenizer,
    sample_data,
    checkpoint_step=None,
    output_dir=None
):
    """
    Test if model can perceive multiple images.

    Tests:
    1. Caption each image individually
    2. Ask to describe "Image 1", "Image 2", etc.
    3. Compare descriptions to see if model differentiates
    """
    print("\n" + "="*80)
    print("MULTI-IMAGE PERCEPTION TEST")
    if checkpoint_step:
        print(f"Checkpoint: step {checkpoint_step}")
    print("="*80)

    # Get images from sample
    images = sample_data['image']  # [N, H, W, 3]
    num_images = sample_data.get('num_images', len(images))
    question = sample_data.get('question', 'N/A')
    answer = sample_data.get('answer', 'N/A')
    sample_id = sample_data.get('sample_id', 'unknown')

    # Save images if output_dir provided
    saved_image_paths = []
    if output_dir:
        image_dir = output_dir / "images" / sample_id
        image_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_images):
            img = images[i]
            # Convert from [-1, 1] to [0, 255]
            img_uint8 = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)

            img_path = image_dir / f"image_{i+1:02d}.png"
            pil_img.save(img_path)
            saved_image_paths.append(str(img_path))

        print(f"\nSaved {num_images} images to: {image_dir}")

    print(f"\nSample Info:")
    print(f"  Number of images: {num_images}")
    print(f"  Question: {question[:100]}...")
    print(f"  Ground truth answer: {answer}")

    # Prepare multi-image batch (same as training/eval)
    # Pad to max 6 images
    max_images = 6
    if num_images < max_images:
        padding = np.zeros((max_images - num_images, *images.shape[1:]), dtype=images.dtype)
        images_padded = np.concatenate([images, padding], axis=0)
    else:
        images_padded = images[:max_images]

    # Stack into batch format [B=1, T=6, H, W, 3]
    image_batch = images_padded[None, ...]

    # Test 1: Individual image captions (single image at a time)
    print(f"\n{'='*80}")
    print("TEST 1: Individual Image Captions")
    print(f"{'='*80}")
    print("Testing each image individually to see if model generates different captions...")

    individual_captions = []
    for i in range(num_images):
        # Create single-image batch (only image i, rest are zeros)
        single_img_padded = np.zeros((max_images, *images.shape[1:]), dtype=np.float32)
        single_img_padded[0] = images[i]  # Put image i in first slot
        single_img_batch = single_img_padded[None, ...]  # [1, 6, H, W, 3]

        caption_prompt = "Describe this image in detail."
        batch, _ = prepare_inference_batch(
            prompt=caption_prompt,
            tokenizer=tokenizer,
            num_images=1,  # Only 1 image
            image_batch=single_img_batch,
            image_size=sample_data.get('image_size', 224),
            max_seq_len=512,
        )

        tokens = decode_fn(
            {"params": params},
            batch=batch,
            max_decode_len=128,
            sampler="greedy",
        )
        tokens = jax.device_get(tokens)
        caption = postprocess_tokens(tokens[0], tokenizer).strip()
        individual_captions.append(caption)

        print(f"\n  Image {i+1}: '{caption[:100]}{'...' if len(caption) > 100 else ''}'")

    # Check diversity
    unique_captions = len(set(individual_captions))
    print(f"\n  Summary: {unique_captions}/{num_images} unique captions")
    if unique_captions == 1:
        print(f"  ⚠️  WARNING: All images have identical captions!")
    elif unique_captions == num_images:
        print(f"  ✓ All images have different captions")
    else:
        print(f"  ⚠️  Some images share the same caption")

    # Test 2A: Progressive Image Count Test (1 to N images)
    print(f"\n{'='*80}")
    print("TEST 2A: Progressive Image Count")
    print(f"{'='*80}")
    print("Testing if model can count images as we add more...")

    count_responses = []
    count_prompt = "How many images do you see? Answer with just the number."

    for i in range(1, num_images + 1):
        # Create batch with i images
        progressive_padded = np.zeros((max_images, *images.shape[1:]), dtype=np.float32)
        for j in range(i):
            progressive_padded[j] = images[j]
        progressive_batch = progressive_padded[None, ...]  # [1, 6, H, W, 3]

        batch, _ = prepare_inference_batch(
            prompt=count_prompt,
            tokenizer=tokenizer,
            num_images=i,  # Tell model there are i images
            image_batch=progressive_batch,
            image_size=sample_data.get('image_size', 224),
            max_seq_len=512,
        )

        tokens = decode_fn(
            {"params": params},
            batch=batch,
            max_decode_len=16,
            sampler="greedy",
        )
        tokens = jax.device_get(tokens)
        count_response = postprocess_tokens(tokens[0], tokenizer).strip()
        count_responses.append(count_response)

        # Check if correct
        import re
        numbers = re.findall(r'\d+', count_response)
        predicted = int(numbers[0]) if numbers else -1
        is_correct = predicted == i
        status = "✓" if is_correct else "✗"

        print(f"  {i} image(s): '{count_response}' {status}")

    # Summary
    import re
    correct_count = 0
    always_one = True
    for i, resp in enumerate(count_responses, 1):
        numbers = re.findall(r'\d+', resp)
        if numbers:
            predicted = int(numbers[0])
            if predicted == i:
                correct_count += 1
            if predicted != 1:
                always_one = False

    print(f"\n  Summary: {correct_count}/{num_images} correct")
    if always_one and num_images > 1:
        print(f"  ✗ CRITICAL: Model always responds '1' regardless of image count!")
    elif correct_count == num_images:
        print(f"  ✓ Model correctly counts all image configurations")

    # Test 2B: Sequential description of all images
    print(f"\n{'='*80}")
    print("TEST 2B: Sequential Multi-Image Description")
    print(f"{'='*80}")

    # Ask model to describe all images sequentially in one response
    prompt = f"You are shown {num_images} images. Please describe each image in order from Image 1 to Image {num_images}. For each image, describe what objects you see, their positions, colors, and what is happening."

    batch, prompt_len = prepare_inference_batch(
        prompt=prompt,
        tokenizer=tokenizer,
        num_images=num_images,  # All images in context
        image_batch=image_batch,
        image_size=sample_data.get('image_size', 224),
        max_seq_len=512,
    )

    max_gen_len = 512  # Much longer for describing all images
    tokens = decode_fn(
        {"params": params},
        batch=batch,
        max_decode_len=max_gen_len,
        sampler="greedy",
    )

    # Decode using postprocess_tokens (same as validation)
    tokens = jax.device_get(tokens)
    sequential_description = postprocess_tokens(tokens[0], tokenizer).strip()

    print(f"\nPrompt: '{prompt}'")
    print(f"  Response:\n{sequential_description}")

    # Also store this for later analysis
    multi_image_captions = [sequential_description]

    # Test 2C: Same Image vs Different Images Comparison
    print(f"\n{'='*80}")
    print("TEST 2C: Same Image vs Different Images (Output Sensitivity Test)")
    print(f"{'='*80}")
    print("\nThis test checks if model output changes based on image content:")
    print("  - Single image (1 image)")
    print("  - Same image repeated 6 times")
    print("  - All different images (original)")
    print("If outputs are identical, the model may not be using image content.\n")

    sensitivity_results = {}
    test_prompt = "Describe what you see in these images. Focus on the visual content."

    # Test A: Single image (just image 0)
    print("  [A] Single image (Image 1 only):")
    single_img = images_padded[0:1]  # [1, H, W, 3]
    padded_single = np.zeros((max_images, *single_img.shape[1:]), dtype=np.float32)
    padded_single[0] = single_img[0]
    single_batch = padded_single[None, ...]  # [1, 6, H, W, 3]

    batch_a, _ = prepare_inference_batch(
        prompt=test_prompt,
        tokenizer=tokenizer,
        num_images=1,
        image_batch=single_batch,
        image_size=sample_data.get('image_size', 224),
        max_seq_len=512,
    )
    tokens_a = decode_fn({"params": params}, batch=batch_a, max_decode_len=128, sampler="greedy")
    response_a = postprocess_tokens(jax.device_get(tokens_a)[0], tokenizer).strip()
    sensitivity_results['single_image'] = response_a
    print(f"      Full Response: '{response_a}'")
    print(f"      Response length: {len(response_a)} chars, {len(response_a.split())} words")

    # Test B: Same image repeated 6 times
    print("\n  [B] Same image repeated 6 times:")
    same_img = images_padded[0]  # [H, W, 3]
    repeated_images = np.stack([same_img] * max_images, axis=0)  # [6, H, W, 3]
    repeated_batch = repeated_images[None, ...]  # [1, 6, H, W, 3]

    batch_b, _ = prepare_inference_batch(
        prompt=test_prompt,
        tokenizer=tokenizer,
        num_images=6,  # Tell model there are 6 images
        image_batch=repeated_batch,
        image_size=sample_data.get('image_size', 224),
        max_seq_len=512,
    )
    tokens_b = decode_fn({"params": params}, batch=batch_b, max_decode_len=128, sampler="greedy")
    response_b = postprocess_tokens(jax.device_get(tokens_b)[0], tokenizer).strip()
    sensitivity_results['same_image_6x'] = response_b
    print(f"      Full Response: '{response_b}'")
    print(f"      Response length: {len(response_b)} chars, {len(response_b.split())} words")

    # Test C: All different images (original)
    print(f"\n  [C] All different images ({num_images} unique images):")
    batch_c, _ = prepare_inference_batch(
        prompt=test_prompt,
        tokenizer=tokenizer,
        num_images=num_images,
        image_batch=image_batch,
        image_size=sample_data.get('image_size', 224),
        max_seq_len=512,
    )
    tokens_c = decode_fn({"params": params}, batch=batch_c, max_decode_len=128, sampler="greedy")
    response_c = postprocess_tokens(jax.device_get(tokens_c)[0], tokenizer).strip()
    sensitivity_results['different_images'] = response_c
    print(f"      Full Response: '{response_c}'")
    print(f"      Response length: {len(response_c)} chars, {len(response_c.split())} words")

    # Test 2D: Logit-level comparison for deeper analysis
    print(f"\n  [D] Logit-Level Comparison (First Token Logits):")
    print("      Comparing raw model outputs before decoding...")

    # Get first-token logits for each configuration
    from src.model import get_first_token_logits, visualize_attention_mask, visualize_attention_weights
    try:
        logits_a = get_first_token_logits(model, params, batch_a)
        logits_b = get_first_token_logits(model, params, batch_b)
        logits_c = get_first_token_logits(model, params, batch_c)

        # Compute differences
        diff_ab = np.abs(logits_a - logits_b).mean()
        diff_bc = np.abs(logits_b - logits_c).mean()
        diff_ac = np.abs(logits_a - logits_c).mean()

        # Also check max diff and if they're exactly identical
        max_diff_ab = np.abs(logits_a - logits_b).max()
        max_diff_bc = np.abs(logits_b - logits_c).max()
        max_diff_ac = np.abs(logits_a - logits_c).max()

        exactly_same_ab = np.allclose(logits_a, logits_b, rtol=1e-5, atol=1e-5)
        exactly_same_bc = np.allclose(logits_b, logits_c, rtol=1e-5, atol=1e-5)
        exactly_same_ac = np.allclose(logits_a, logits_c, rtol=1e-5, atol=1e-5)

        print(f"      Single vs Same×6:    mean_diff={diff_ab:.6f}, max_diff={max_diff_ab:.6f}, exact_match={exactly_same_ab}")
        print(f"      Same×6 vs Different: mean_diff={diff_bc:.6f}, max_diff={max_diff_bc:.6f}, exact_match={exactly_same_bc}")
        print(f"      Single vs Different: mean_diff={diff_ac:.6f}, max_diff={max_diff_ac:.6f}, exact_match={exactly_same_ac}")

        sensitivity_results['logit_analysis'] = {
            'single_vs_same6': {'mean_diff': float(diff_ab), 'max_diff': float(max_diff_ab), 'exact_match': bool(exactly_same_ab)},
            'same6_vs_different': {'mean_diff': float(diff_bc), 'max_diff': float(max_diff_bc), 'exact_match': bool(exactly_same_bc)},
            'single_vs_different': {'mean_diff': float(diff_ac), 'max_diff': float(max_diff_ac), 'exact_match': bool(exactly_same_ac)},
        }

        if exactly_same_bc:
            print(f"\n      ✗ CRITICAL: Same×6 and Different have IDENTICAL logits!")
            print(f"        This means the model is NOT seeing the image content differences at all.")
            print(f"        The problem is likely in:")
            print(f"          - Image preprocessing (all images normalized to same values?)")
            print(f"          - ViT forward pass (not processing images correctly?)")
            print(f"          - Image embedding injection into LLM")
        elif max_diff_bc < 0.01:
            print(f"\n      ⚠️  WARNING: Logit differences are very small (max < 0.01)")
            print(f"        The model barely distinguishes between Same×6 and Different images.")
        else:
            print(f"\n      ✓ Logits differ meaningfully between image configurations")

    except Exception as e:
        print(f"      ⚠️  Could not compute logits: {e}")
        print(f"        (get_first_token_logits may need to be implemented in src/model.py)")
        sensitivity_results['logit_analysis'] = {'error': str(e)}

    # Test 2E: Attention Visualization (REAL Attention Weights)
    print(f"\n  [E] REAL Attention Weight Visualization:")
    print("      (Using actual softmax attention probabilities from last transformer layer)")
    try:
        if output_dir:
            attn_dir = output_dir / "attention_viz" / sample_id
            attn_dir.mkdir(parents=True, exist_ok=True)

            # Visualize attention for each configuration
            print("      Generating attention visualizations...")

            # REAL attention visualization from the last layer (layer 17 in Gemma 2B)
            attn_info_a = visualize_attention_weights(
                model, params, batch_a,
                output_path=attn_dir / "attention_single_image.png",
                title="Single Image (1 image)",
                layer_idx=-1  # Last layer
            )
            attn_info_b = visualize_attention_weights(
                model, params, batch_b,
                output_path=attn_dir / "attention_same_6x.png",
                title="Same Image ×6",
                layer_idx=-1
            )
            attn_info_c = visualize_attention_weights(
                model, params, batch_c,
                output_path=attn_dir / "attention_different.png",
                title=f"Different Images ({num_images} unique)",
                layer_idx=-1
            )

            print(f"      Single image:  {attn_info_a['num_image_tokens']} img tokens ({attn_info_a['num_images']} images)")
            print(f"      Same×6:        {attn_info_b['num_image_tokens']} img tokens ({attn_info_b['num_images']} images)")
            print(f"      Different:     {attn_info_c['num_image_tokens']} img tokens ({attn_info_c['num_images']} images)")
            print(f"      Text→Image attention: A={attn_info_a['text_to_image_attention']:.4f}, B={attn_info_b['text_to_image_attention']:.4f}, C={attn_info_c['text_to_image_attention']:.4f}")
            print(f"      Text→Text attention:  A={attn_info_a['text_to_text_attention']:.4f}, B={attn_info_b['text_to_text_attention']:.4f}, C={attn_info_c['text_to_text_attention']:.4f}")

            sensitivity_results['attention_info'] = {
                'single_image': attn_info_a,
                'same_6x': attn_info_b,
                'different': attn_info_c,
            }
        else:
            print("      Skipped (no output_dir provided)")
    except Exception as e:
        print(f"      ⚠️  Could not visualize attention: {e}")
        import traceback
        traceback.print_exc()

    # Analysis of sensitivity (text output comparison)
    print(f"\n  --- Text Output Sensitivity Analysis ---")
    a_b_same = response_a.strip() == response_b.strip()
    b_c_same = response_b.strip() == response_c.strip()
    a_c_same = response_a.strip() == response_c.strip()
    all_same = a_b_same and b_c_same

    print(f"  Single vs Same×6:     {'IDENTICAL ⚠️' if a_b_same else 'Different ✓'}")
    print(f"  Same×6 vs Different:  {'IDENTICAL ⚠️' if b_c_same else 'Different ✓'}")
    print(f"  Single vs Different:  {'IDENTICAL ⚠️' if a_c_same else 'Different ✓'}")

    if all_same:
        print(f"\n  ✗ CRITICAL: All outputs are IDENTICAL!")
        print(f"    The model is NOT responding to image content changes.")
        print(f"    Possible causes:")
        print(f"      1. Images not reaching the model properly")
        print(f"      2. Image embeddings not affecting LLM output")
        print(f"      3. Model has collapsed to text-only behavior")
    elif a_b_same and not b_c_same:
        print(f"\n  ⚠️  WARNING: Single and Same×6 produce identical output")
        print(f"    Model may not be using image count information correctly")
    elif not a_b_same and not b_c_same and not a_c_same:
        print(f"\n  ✓ GOOD: Model produces different outputs for different image configurations")
        print(f"    This suggests the model IS responding to image content")

    # Test 3: Answer the actual question
    print(f"\n{'='*80}")
    print("TEST 3: Answer Original Question")
    print(f"{'='*80}")

    # Use the SAME tokenization as training
    batch, prompt_len = prepare_inference_batch(
        prompt=question,
        tokenizer=tokenizer,
        num_images=num_images,
        image_batch=image_batch,
        image_size=sample_data.get('image_size', 224),
        max_seq_len=1024,  # Longer for complex questions
    )

    max_gen_len = 64
    tokens = decode_fn(
        {"params": params},
        batch=batch,
        max_decode_len=max_gen_len,
        sampler="greedy",
    )

    # Decode using postprocess_tokens (same as validation)
    tokens = jax.device_get(tokens)
    model_answer = postprocess_tokens(tokens[0], tokenizer).strip()

    print(f"\nQuestion: {question[:200]}...")  # Truncate for display
    print(f"  Model answer: {model_answer}")
    print(f"  Ground truth: {answer}")
    print(f"  Correct: {model_answer.lower().strip() == answer.lower().strip()}")

    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    # Check if individual captions are different
    unique_captions = len(set(individual_captions))
    print(f"\n1. Individual Caption Diversity (TEST 1):")
    print(f"   Unique captions: {unique_captions}/{num_images}")
    if unique_captions == 1:
        print(f"   ⚠️  WARNING: All images have the same caption!")
        print(f"   This suggests the model may not be differentiating between images.")
    else:
        print(f"   ✓ Model generates different captions for different images")

    # Check progressive image count accuracy
    print(f"\n2. Progressive Image Count Accuracy (TEST 2A):")
    # if count_responses:
    import re
    correct_counts = 0
    for i, response in enumerate(count_responses, 1):
        numbers = re.findall(r'\d+', response)
        if numbers and int(numbers[0]) == i:
            correct_counts += 1

    accuracy = correct_counts / len(count_responses) * 100
    print(f"   Accuracy: {correct_counts}/{len(count_responses)} correct ({accuracy:.1f}%)")

    if accuracy == 100:
        print(f"   ✓ Model correctly counted images at all levels")
    elif accuracy == 0:
        print(f"   ✗ CRITICAL: Model failed to count ANY image configuration correctly")
        print(f"   This suggests the model may NOT be receiving multiple images properly")
    else:
        print(f"   ⚠️  WARNING: Model only counted correctly {accuracy:.0f}% of the time")

    # Check if model always says "1"
    always_one = all(re.findall(r'\d+', resp) and int(re.findall(r'\d+', resp)[0]) == 1 for resp in count_responses)
    if always_one:
        print(f"   ✗ CRITICAL: Model ALWAYS responds '1' regardless of actual image count")
        print(f"   This strongly indicates multi-image input is NOT working")
    # else:
    #     print(f"   SKIPPED (test disabled)")

    # Check sequential description
    print(f"\n3. Sequential Description (TEST 2B):")
    seq_desc_len = len(sequential_description)
    print(f"   Response length: {seq_desc_len} characters")
    if seq_desc_len < 50:
        print(f"   ⚠️  WARNING: Very short response - model may not be describing all images")
    else:
        # Check if response mentions multiple images
        image_mentions = sum(1 for i in range(1, num_images + 1) if f"image {i}" in sequential_description.lower() or f"image{i}" in sequential_description.lower())
        print(f"   Image references found: {image_mentions}/{num_images}")
        if image_mentions >= num_images - 1:
            print(f"   ✓ Model appears to address most/all images")
        else:
            print(f"   ⚠️  WARNING: Model may not be describing all images separately")

    print(f"\n4. Question Answering (TEST 3):")
    is_correct = model_answer.lower().strip() == answer.lower().strip()
    if is_correct:
        print(f"   ✓ Correct answer: '{model_answer}'")
    else:
        print(f"   ✗ Incorrect answer")
        print(f"     Model: '{model_answer}'")
        print(f"     Expected: '{answer}'")

    # Return results for further analysis
    return {
        'sample_id': sample_id,
        'num_images': num_images,
        'question': question,
        'ground_truth': answer,
        'individual_captions': individual_captions,
        'count_responses': count_responses,  # All progressive count responses
        'count_response': count_response,  # Final count response
        'sequential_description': sequential_description,
        'sensitivity_test': {
            'single_image': sensitivity_results.get('single_image', ''),
            'same_image_6x': sensitivity_results.get('same_image_6x', ''),
            'different_images': sensitivity_results.get('different_images', ''),
            'single_vs_same6_identical': a_b_same,
            'same6_vs_different_identical': b_c_same,
            'single_vs_different_identical': a_c_same,
            'all_identical': all_same,
        },
        'model_answer': model_answer,
        'correct': is_correct,
        'image_paths': saved_image_paths if saved_image_paths else None,
    }


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.npz"))

    if not checkpoints:
        return None

    # Sort by step number
    checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
    return checkpoints[-1]


def main():
    parser = argparse.ArgumentParser(description="Test multi-image perception")
    parser.add_argument("--env", type=str, required=True, help="Path to .env config file")
    parser.add_argument("--checkpoint", type=str, default="from_env",
                       help="Checkpoint to use: 'from_env' (use MODEL_CHECKPOINT_PATH from .env), 'latest' (latest trained), or path to .npz file")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to test (default: 0)")
    parser.add_argument("--sample_id", type=str, default=None, help="Specific sample ID to test")

    args = parser.parse_args()

    # Load config
    config = load_config(args.env)

    print("Loading model...")
    model, base_params, tokenizer, decode_fn = load_paligemma_model(config)

    # Debug: Check params structure
    print(f"\nDebug: Checking params structure...")
    print(f"  Top-level keys: {list(base_params.keys())}")
    if 'img' in base_params:
        print(f"  'img' keys: {list(base_params['img'].keys())}")
    else:
        print(f"  WARNING: 'img' key not found in params!")

    # Determine checkpoint to use
    if args.checkpoint == "from_env":
        # Use the base model from .env (MODEL_CHECKPOINT_PATH)
        print(f"Using base model from .env: {config.model.checkpoint_path}")
        params = base_params
        checkpoint_step = "base"
    elif args.checkpoint == "latest":
        # Auto-detect latest trained checkpoint
        checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
        if checkpoint_path:
            print(f"Auto-detected latest checkpoint: {checkpoint_path}")
            params = load_checkpoint(checkpoint_path, model, base_params, config)
            checkpoint_step = checkpoint_path.stem.split('_')[-1]
        else:
            print("No checkpoint found, using base pretrained model from .env")
            params = base_params
            checkpoint_step = "base"
    else:
        # Specific checkpoint path provided
        print(f"Loading checkpoint: {args.checkpoint}")
        params = load_checkpoint(args.checkpoint, model, base_params, config)
        checkpoint_step = Path(args.checkpoint).stem.split('_')[-1]

    # Load dataset
    print("\nLoading dataset...")

    # Read JSONL file directly to get sample
    jsonl_path = os.path.join(config.data.base_dir, config.data.valid_file)
    image_base_dir = os.path.join(config.data.base_dir, config.data.image_dir)

    # Read all samples
    with open(jsonl_path, 'r') as f:
        all_samples = [json.loads(line) for line in f]

    # Get target sample
    if args.sample_id:
        target_sample = None
        for sample in all_samples:
            if sample.get('sample_id') == args.sample_id:
                target_sample = sample
                break
        if target_sample is None:
            print(f"Error: Sample ID '{args.sample_id}' not found")
            return
    else:
        if args.sample_idx >= len(all_samples):
            print(f"Error: Sample index {args.sample_idx} out of range (max: {len(all_samples)-1})")
            return
        target_sample = all_samples[args.sample_idx]

    # Process the sample manually (load images)
    from src.data import preprocess_image
    from PIL import Image as PILImage

    sample_id = target_sample.get('sample_id', f'sample_{args.sample_idx}')

    # Extract question from prompt_blocks
    prompt_blocks = target_sample['prompt_blocks']
    question_text = ""
    image_urls = []

    for block in prompt_blocks:
        if block['type'] == 'text' and block.get('text'):
            question_text += block['text']
        elif block['type'] == 'image_url' and block.get('image_url'):
            # image_url is nested: {"url": "images/xxx.png"}
            img_url = block['image_url']['url']
            image_urls.append(img_url)

    answer = target_sample['ground_truth_answer']

    # Load images
    images = []
    for img_url in image_urls:
        # img_url is like "images/xxx.png"
        full_path = os.path.join(config.data.base_dir, img_url)
        pil_img = PILImage.open(full_path).convert('RGB')
        img = preprocess_image(pil_img, size=config.model.img_size)
        images.append(img)

    # Stack images into array [N, H, W, 3]
    images_array = np.stack(images, axis=0)

    sample_data = {
        'sample_id': sample_id,
        'question': question_text,
        'answer': answer,
        'image': images_array,
        'num_images': len(images),
        'image_size': config.model.img_size,  # Add image size for tokenization
    }

    print(f"\nTesting sample: {sample_data.get('sample_id', args.sample_idx)}")

    # Prepare output directory
    output_dir = Path(config.checkpoint_dir) / "perception_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run test (with output_dir for saving images)
    results = test_multi_image_perception(
        model, params, decode_fn, tokenizer,
        sample_data,
        checkpoint_step=checkpoint_step,
        output_dir=output_dir
    )

    results_file = output_dir / f"test_step_{checkpoint_step}_sample_{sample_data.get('sample_id', args.sample_idx)}.json"

    # Results already have all needed fields from test_multi_image_perception
    results_serializable = {
        'checkpoint_step': checkpoint_step,
        **results  # Unpack all results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

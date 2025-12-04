#!/usr/bin/env python3
"""
Compare PaliGemma Checkpoints: Vanilla vs OpenPI pi0.5_base

This script compares PaliGemma checkpoints from different sources:
1. Vanilla PaliGemma: gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz
2. OpenPI pi0.5_base: gs://openpi-assets/checkpoints/pi05_base (pre-trained on 10k+ hours robot data)

The comparison includes:
- Parameter shape comparison
- Parameter value comparison (are they identical or different?)
- XVR eval accuracy comparison

Usage:
    # Default: Compare vanilla PaliGemma vs pi0.5_base
    python tests/compare_checkpoints.py
    python tests/compare_checkpoints.py --num-examples 50
    python tests/compare_checkpoints.py --skip-eval  # Only compare parameters
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np


def print_banner(text: str):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def download_vanilla_paligemma(target_path: str) -> str:
    """Download vanilla PaliGemma checkpoint from Google Cloud."""
    target_path = Path(target_path)

    if target_path.exists():
        print(f"  Vanilla PaliGemma already exists: {target_path}")
        return str(target_path)

    url = "https://storage.googleapis.com/vertex-model-garden-paligemma-us/paligemma/pt_224.npz"
    print(f"  Downloading vanilla PaliGemma from: {url}")

    import requests
    from tqdm import tqdm

    target_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(target_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading Vanilla") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

    print(f"  Downloaded: {target_path}")
    return str(target_path)


def download_pi05_base_checkpoint(cache_dir: str) -> str:
    """Download OpenPI pi0.5_base checkpoint and extract PaliGemma weights.

    Tries multiple methods:
    1. Check if already extracted in project root (from setup.sh)
    2. Check if already in cache
    3. Try using openpi's download utilities
    4. Fall back to gsutil command
    """
    import flax.traverse_util
    import subprocess

    cache_dir = Path(cache_dir)
    extracted_path = cache_dir / "pi05_base_paligemma.npz"

    # Method 1: Check project root (setup.sh puts it there)
    project_root_path = PROJECT_ROOT / "pi05_base_paligemma.npz"
    if project_root_path.exists():
        print(f"  pi0.5_base PaliGemma weights found in project root: {project_root_path}")
        return str(project_root_path)

    # Method 2: Check cache
    if extracted_path.exists():
        print(f"  pi0.5_base PaliGemma weights already extracted: {extracted_path}")
        return str(extracted_path)

    print("  Downloading OpenPI pi0.5_base checkpoint...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Method 3: Try openpi's download utilities
    openpi_path = PROJECT_ROOT.parent / "openpi"
    params_path = None

    if openpi_path.exists():
        try:
            sys.path.insert(0, str(openpi_path / "src"))
            # Only import download, not model (which imports TensorFlow and causes conflicts)
            import openpi.shared.download as download

            params_path = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base/params")
            print(f"  Downloaded to cache: {params_path}")
        except Exception as e:
            print(f"  OpenPI import/download failed: {e}")
            # Don't print full traceback for TensorFlow conflicts
            if "DType" not in str(e):
                import traceback
                traceback.print_exc()
            print("  Trying gsutil fallback...")

    # Method 4: Fall back to gsutil
    if params_path is None:
        gs_url = "gs://openpi-assets/checkpoints/pi05_base/params"
        local_params_dir = cache_dir / "pi05_base_params"
        local_params_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading via gsutil from {gs_url}...")
        try:
            # Try to find gsutil in common locations
            gsutil_cmd = "gsutil"
            import shutil
            if not shutil.which("gsutil"):
                # Try common installation locations
                possible_paths = [
                    Path.home() / "google-cloud-sdk" / "bin" / "gsutil",
                    Path("/usr/local/bin/gsutil"),
                    Path("/usr/bin/gsutil"),
                ]
                for path in possible_paths:
                    if path.exists():
                        gsutil_cmd = str(path)
                        break
                else:
                    raise FileNotFoundError("gsutil not found in PATH or common locations")
            
            # Use rsync instead of cp -r for better directory handling
            subprocess.run(
                [gsutil_cmd, "-m", "rsync", "-r", gs_url, str(local_params_dir)],
                check=True,
                capture_output=True,
                text=True
            )
            params_path = local_params_dir
            print(f"  Downloaded to: {params_path}")
        except subprocess.CalledProcessError as e:
            print(f"  gsutil failed: {e.stderr}")
            raise RuntimeError(
                "Could not download pi0.5_base checkpoint.\n"
                "Please run setup.sh first, or install gsutil and try again.\n"
                "Alternatively, manually download from:\n"
                "  gs://openpi-assets/checkpoints/pi05_base/params"
            )
        except FileNotFoundError:
            raise RuntimeError(
                "gsutil not found. Please either:\n"
                "1. Run setup.sh first to download the checkpoint, or\n"
                "2. Install Google Cloud SDK (which includes gsutil)"
            )

    # Load and extract PaliGemma weights
    print("  Extracting PaliGemma weights from pi0.5_base...")

    # Try to load using openpi's restore_params if available
    params = None
    try:
        # Try to import and use restore_params without importing the full model module
        # (which imports TensorFlow and causes conflicts)
        if openpi_path.exists():
            try:
                sys.path.insert(0, str(openpi_path / "src"))
                # Import directly from the model file to avoid PyTorch imports
                from openpi.models import model as openpi_model
                params = openpi_model.restore_params(str(params_path), restore_type=np.ndarray)
                print("  Loaded using openpi.restore_params")
            except Exception as e:
                if "DType" not in str(e):
                    print(f"  Failed to use openpi.restore_params: {e}")
    except Exception:
        pass
    
    # Fall back to manual loading using orbax
    if params is None:
        try:
            import orbax.checkpoint as ocp
            import jax
            from flax import traverse_util
            
            print("  Loading using orbax.PyTreeCheckpointer...")
            ckptr = ocp.PyTreeCheckpointer()
            params_path_str = str(params_path)
            
            # Get metadata to understand structure
            metadata = ckptr.metadata(params_path_str)
            print(f"  Metadata keys: {list(metadata.keys())}")
            
            # Construct restore item
            if "params" not in metadata:
                raise KeyError(f"'params' not found in metadata. Available keys: {list(metadata.keys())}")
            
            item = {"params": metadata["params"]}
            
            # Try restoring without restore_args first
            try:
                params = ckptr.restore(
                    params_path_str,
                    ocp.args.PyTreeRestore(
                        item=item,
                        restore_args=None,
                    ),
                )["params"]
                print("  Restored without restore_args")
            except Exception as e1:
                print(f"  Restore without restore_args failed: {e1}")
                print("  Trying with ArrayRestoreArgs...")
                # Create restore args for numpy arrays
                restore_args = jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray),
                    item
                )
                params = ckptr.restore(
                    params_path_str,
                    ocp.args.PyTreeRestore(
                        item=item,
                        restore_args=restore_args,
                    ),
                )["params"]
            
            # Remove "value" suffix if present (from nnx.State)
            flat_params = traverse_util.flatten_dict(params)
            if flat_params and all(kp[-1] == "value" for kp in flat_params):
                flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
                params = traverse_util.unflatten_dict(flat_params)
        except Exception as e:
            print(f"  Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Could not load checkpoint from {params_path}: {e}")

    # Check structure
    print(f"  Top-level keys: {list(params.keys())}")

    # Extract only PaliGemma part
    if "PaliGemma" in params:
        paligemma_params = params["PaliGemma"]

        # Flatten to match vanilla PaliGemma format (params/llm/..., params/img/...)
        flat_params = flax.traverse_util.flatten_dict({"params": paligemma_params}, sep="/")

        # Save extracted weights
        np.savez(extracted_path, **flat_params)
        print(f"  Extracted {len(flat_params)} parameter arrays")
        print(f"  Saved to: {extracted_path}")
        return str(extracted_path)
    else:
        print(f"  Warning: 'PaliGemma' key not found in pi0.5_base params")
        print(f"  Available keys: {list(params.keys())}")
        raise KeyError("PaliGemma not found in pi0.5_base")


def compare_two_checkpoints(path_a: str, name_a: str, path_b: str, name_b: str) -> dict:
    """Compare parameters between two checkpoints (memory-efficient with mmap)."""
    print(f"\n--- Comparing {name_a} vs {name_b} ---\n")

    # Use mmap mode for memory-efficient loading
    print(f"Opening {name_a} (mmap mode)...")
    params_a = np.load(path_a, mmap_mode='r', allow_pickle=False)
    keys_a = set(params_a.files)
    print(f"  Found {len(keys_a)} parameter arrays")

    print(f"Opening {name_b} (mmap mode)...")
    params_b = np.load(path_b, mmap_mode='r', allow_pickle=False)
    keys_b = set(params_b.files)
    print(f"  Found {len(keys_b)} parameter arrays")

    print(f"\n{name_a} keys: {len(keys_a)}")
    print(f"{name_b} keys: {len(keys_b)}")

    only_in_a = keys_a - keys_b
    only_in_b = keys_b - keys_a
    common_keys = keys_a & keys_b

    print(f"Common keys: {len(common_keys)}")
    print(f"Only in {name_a}: {len(only_in_a)}")
    print(f"Only in {name_b}: {len(only_in_b)}")

    if only_in_a:
        print(f"\n  Keys only in {name_a} (first 5):")
        for k in sorted(only_in_a)[:5]:
            print(f"    - {k}")

    if only_in_b:
        print(f"\n  Keys only in {name_b} (first 5):")
        for k in sorted(only_in_b)[:5]:
            print(f"    - {k}")

    # Compare shapes and values (lazy loading)
    shape_mismatches = []
    value_mismatches = []
    identical_params = []

    print(f"\nComparing {len(common_keys)} common parameters...")
    for i, key in enumerate(sorted(common_keys)):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(common_keys)}")
        
        arr_a = params_a[key]
        arr_b = params_b[key]

        if arr_a.shape != arr_b.shape:
            shape_mismatches.append((key, arr_a.shape, arr_b.shape))
        else:
            # Load into memory for comparison (one at a time)
            a_data = np.array(arr_a)
            b_data = np.array(arr_b)
            
            # Compare values
            if np.allclose(a_data, b_data, rtol=1e-5, atol=1e-8):
                identical_params.append(key)
            else:
                max_diff = np.max(np.abs(a_data.astype(np.float32) - b_data.astype(np.float32)))
                value_mismatches.append((key, max_diff))
            
            # Free memory
            del a_data, b_data

    print(f"\nIdentical parameters: {len(identical_params)} / {len(common_keys)}")
    print(f"Shape mismatches: {len(shape_mismatches)}")
    print(f"Value mismatches: {len(value_mismatches)}")

    if value_mismatches:
        print(f"\n  Value mismatches (top 10 by max diff):")
        for key, max_diff in sorted(value_mismatches, key=lambda x: -x[1])[:10]:
            print(f"    {key}: max diff = {max_diff:.6e}")

    # Skip hash computation for large files (too slow with mmap)
    hashes_match = len(value_mismatches) == 0 and len(shape_mismatches) == 0

    if hashes_match:
        print(f"\n*** All {len(common_keys)} common parameters are IDENTICAL ***")
    else:
        print(f"\n*** Parameters are DIFFERENT ***")

    return {
        "name_a": name_a,
        "name_b": name_b,
        "keys_a": len(keys_a),
        "keys_b": len(keys_b),
        "common_keys": len(common_keys),
        "identical_params": len(identical_params),
        "shape_mismatches": len(shape_mismatches),
        "value_mismatches": len(value_mismatches),
        "hashes_match": hashes_match,
    }


def evaluate_checkpoint(checkpoint_path: str, checkpoint_name: str, config, num_examples: int) -> dict:
    """Evaluate a single checkpoint on XVR eval dataset."""
    import jax
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.set_visible_devices([], "TPU")

    from src.model import load_paligemma_model, create_trainable_mask, prepare_params_for_training
    from src.data import XVRDataset
    from src.evaluation import evaluate_model

    print(f"\n--- Evaluating {checkpoint_name} ---")
    print(f"Checkpoint: {checkpoint_path}")

    # Temporarily override config checkpoint path
    original_path = config.model.checkpoint_path
    original_url = getattr(config.model, 'checkpoint_url', None)
    config.model.checkpoint_path = checkpoint_path
    config.model.checkpoint_url = None  # Don't try to download

    # Load model
    model, params, tokenizer, decode_fn = load_paligemma_model(config)

    # Restore config
    config.model.checkpoint_path = original_path
    config.model.checkpoint_url = original_url

    # Prepare params
    trainable_mask = create_trainable_mask(params, strategy="attention_only", config=config)
    params = prepare_params_for_training(params, trainable_mask, config)

    # Load eval dataset
    eval_dataset = XVRDataset(
        jsonl_path=os.path.join(config.data.base_dir, config.data.eval_file),
        image_base_dir=config.data.base_dir,
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        image_size=config.model.img_size,
    )

    print(f"Evaluating on {num_examples} examples...")

    metrics = evaluate_model(
        model,
        params,
        decode_fn,
        tokenizer,
        eval_dataset,
        config,
        num_examples=num_examples,
        verbose=True,
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare PaliGemma checkpoints")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation, only compare parameters",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=str(PROJECT_ROOT / "envs" / ".env.example"),
        help="Path to .env file for evaluation",
    )

    args = parser.parse_args()

    print_banner("PaliGemma Checkpoint Comparison")
    print("Comparing checkpoints:")
    print("  1. Vanilla PaliGemma (Google Cloud)")
    print("  2. OpenPI pi0.5_base (10k+ hours robot data)")

    # Set default paths
    cache_dir = PROJECT_ROOT / "tests" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {}

    # Download checkpoints
    print_banner("Downloading Checkpoints")

    # Vanilla PaliGemma
    checkpoints['vanilla'] = download_vanilla_paligemma(str(cache_dir / "vanilla_paligemma.npz"))

    # OpenPI pi0.5_base
    try:
        checkpoints['pi05_base'] = download_pi05_base_checkpoint(str(cache_dir))
    except Exception as e:
        print(f"\nError: Could not download pi0.5_base: {e}")
        print("Make sure openpi is cloned at ../openpi")
        return

    # Compare parameters
    print_banner("Parameter Comparison")

    comp = compare_two_checkpoints(
        checkpoints['vanilla'], "Vanilla PaliGemma",
        checkpoints['pi05_base'], "pi0.5_base"
    )

    # Summary
    print_banner("Comparison Summary")

    status = "IDENTICAL" if comp['hashes_match'] else "DIFFERENT"
    identical_pct = comp['identical_params'] / comp['common_keys'] * 100 if comp['common_keys'] > 0 else 0

    print(f"Vanilla PaliGemma vs pi0.5_base: {status}")
    print(f"  - Identical params: {comp['identical_params']}/{comp['common_keys']} ({identical_pct:.1f}%)")
    print(f"  - Value mismatches: {comp['value_mismatches']}")

    if comp['hashes_match']:
        print("\n*** CHECKPOINTS ARE IDENTICAL ***")
        print("pi0.5_base contains the same PaliGemma weights as vanilla.")
    else:
        print("\n*** CHECKPOINTS ARE DIFFERENT ***")
        print("pi0.5_base has been fine-tuned - PaliGemma weights differ from vanilla.")

    # Run evaluation
    if not args.skip_eval:
        print_banner("XVR Evaluation")

        # Load config
        from src.config import load_config
        config = load_config(args.env)

        # Check if eval file exists
        eval_file = os.path.join(config.data.base_dir, config.data.eval_file)
        if not os.path.exists(eval_file):
            print(f"Warning: Eval file not found: {eval_file}")
            print("Skipping evaluation. Please set DATA_EVAL_FILE in your .env")
            return

        import jax
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config.system.xla_mem_fraction)
        print(f"JAX devices: {jax.devices()}")

        results = {}

        for name, path in checkpoints.items():
            display_name = {
                'vanilla': 'Vanilla PaliGemma',
                'pi05_base': 'OpenPI pi0.5_base'
            }.get(name, name)

            try:
                results[name] = evaluate_checkpoint(path, display_name, config, args.num_examples)
            except Exception as e:
                print(f"Error evaluating {display_name}: {e}")
                import traceback
                traceback.print_exc()

        # Print comparison
        print_banner("Evaluation Results Summary")

        print(f"{'Checkpoint':<30} {'Accuracy':<15} {'Correct':<15}")
        print("-" * 60)

        for name in ['vanilla', 'pi05_base']:
            if name in results:
                r = results[name]
                display_name = {
                    'vanilla': 'Vanilla PaliGemma',
                    'pi05_base': 'OpenPI pi0.5_base'
                }.get(name, name)
                print(f"{display_name:<30} {r['accuracy']:.2%}          {r['correct']}/{r['total']}")

        # Accuracy difference
        if 'vanilla' in results and 'pi05_base' in results:
            diff = results['pi05_base']['accuracy'] - results['vanilla']['accuracy']
            print(f"\nAccuracy difference (pi0.5_base - vanilla): {diff:+.2%}")

            if diff > 0.01:
                print("pi0.5_base performs BETTER on XVR eval!")
            elif diff < -0.01:
                print("Vanilla PaliGemma performs BETTER on XVR eval!")
            else:
                print("Performance is similar (within 1%)")


if __name__ == "__main__":
    main()

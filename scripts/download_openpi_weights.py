#!/usr/bin/env python3
"""
Download and convert OpenPI checkpoint to big_vision NPZ format.

This script downloads an OpenPI checkpoint (e.g., pi0_base) and extracts
the PaliGemma weights into a format compatible with big_vision/config_pi_training.

OpenPI checkpoints contain:
- PaliGemma backbone (fine-tuned on robotics data)
- Action Expert (gemma_300m)
- Action projection layers

We extract only the PaliGemma portion and convert it to NPZ format.

Usage:
    python scripts/download_openpi_weights.py
    python scripts/download_openpi_weights.py --checkpoint pi0_fast_base
    python scripts/download_openpi_weights.py --output ./my_paligemma.npz
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add openpi to path
OPENPI_ROOT = PROJECT_ROOT.parent / "openpi"
sys.path.insert(0, str(OPENPI_ROOT / "src"))


def download_openpi_checkpoint(checkpoint_name: str) -> Path:
    """Download OpenPI checkpoint using their download utility."""
    from openpi.shared import download

    checkpoint_url = f"gs://openpi-assets/checkpoints/{checkpoint_name}/params"
    print(f"Downloading OpenPI checkpoint: {checkpoint_url}")

    local_path = download.maybe_download(checkpoint_url)
    print(f"Downloaded to: {local_path}")

    return local_path


def load_openpi_params(params_path: Path) -> dict:
    """Load parameters from OpenPI checkpoint (Orbax format)."""
    import orbax.checkpoint as ocp
    from flax import traverse_util
    import numpy as np

    print(f"Loading parameters from: {params_path}")

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=None,
            ),
        )["params"]

    # Remove "value" suffix if present (from nnx.State)
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    params = traverse_util.unflatten_dict(flat_params)

    return params


def extract_paligemma_weights(params: dict) -> dict:
    """
    Extract PaliGemma weights from OpenPI params.

    OpenPI structure:
        PaliGemma/
            llm/...
            img/...
        action_in_proj/...
        action_out_proj/...
        state_proj/...
        action_time_mlp_in/...
        action_time_mlp_out/...

    We extract only PaliGemma/llm and PaliGemma/img.
    """
    from flax import traverse_util
    import numpy as np

    print("Extracting PaliGemma weights from OpenPI checkpoint...")

    flat_params = traverse_util.flatten_dict(params, sep="/")

    # Filter to only PaliGemma weights
    paligemma_params = {}
    for key, value in flat_params.items():
        if key.startswith("PaliGemma/"):
            # Convert to big_vision format: PaliGemma/llm/... -> params/llm/...
            new_key = key.replace("PaliGemma/", "params/")
            paligemma_params[new_key] = np.array(value)

    print(f"  Extracted {len(paligemma_params)} parameter arrays")

    # Show some sample keys
    sample_keys = sorted(paligemma_params.keys())[:5]
    print(f"  Sample keys: {sample_keys}")

    return paligemma_params


def save_as_npz(params: dict, output_path: Path):
    """Save parameters in NPZ format compatible with big_vision."""
    import numpy as np

    print(f"Saving to NPZ format: {output_path}")

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(output_path, **params)

    # Verify
    file_size = output_path.stat().st_size / (1024**3)
    print(f"  Saved successfully ({file_size:.2f} GB)")


def verify_checkpoint(npz_path: Path):
    """Verify the converted checkpoint."""
    import numpy as np

    print(f"\nVerifying checkpoint: {npz_path}")

    loaded = np.load(npz_path)
    keys = sorted(loaded.files)

    print(f"  Total keys: {len(keys)}")
    print(f"  Sample keys:")
    for k in keys[:10]:
        print(f"    {k}: {loaded[k].shape} ({loaded[k].dtype})")

    # Check for expected components
    has_llm = any("llm" in k for k in keys)
    has_img = any("img" in k for k in keys)

    print(f"\n  Has LLM weights: {has_llm}")
    print(f"  Has Image encoder weights: {has_img}")

    if has_llm and has_img:
        print("\n  Checkpoint is valid for big_vision/PaliGemma!")
        return True
    else:
        print("\n  WARNING: Checkpoint may be incomplete!")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert OpenPI checkpoint for config_pi_training"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pi0_base",
        help="OpenPI checkpoint name (e.g., pi0_base, pi0_fast_base, pi05_base)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for NPZ file (default: ./openpi_{checkpoint}.npz)"
    )
    parser.add_argument(
        "--verify-only",
        type=str,
        default=None,
        help="Only verify an existing NPZ checkpoint"
    )

    args = parser.parse_args()

    # Verify only mode
    if args.verify_only:
        verify_checkpoint(Path(args.verify_only))
        return

    # Set default output path
    if args.output is None:
        args.output = str(PROJECT_ROOT / f"openpi_{args.checkpoint}_paligemma.npz")

    output_path = Path(args.output)

    print("=" * 80)
    print("OpenPI Checkpoint to big_vision NPZ Converter")
    print("=" * 80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Output: {output_path}\n")

    # Check if output already exists
    if output_path.exists():
        print(f"Output file already exists: {output_path}")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    try:
        # Step 1: Download checkpoint
        print("\n" + "=" * 80)
        print("Step 1: Downloading OpenPI checkpoint")
        print("=" * 80 + "\n")
        params_path = download_openpi_checkpoint(args.checkpoint)

        # Step 2: Load parameters
        print("\n" + "=" * 80)
        print("Step 2: Loading parameters")
        print("=" * 80 + "\n")
        params = load_openpi_params(params_path)

        # Step 3: Extract PaliGemma weights
        print("\n" + "=" * 80)
        print("Step 3: Extracting PaliGemma weights")
        print("=" * 80 + "\n")
        paligemma_params = extract_paligemma_weights(params)

        # Step 4: Save as NPZ
        print("\n" + "=" * 80)
        print("Step 4: Saving as NPZ")
        print("=" * 80 + "\n")
        save_as_npz(paligemma_params, output_path)

        # Step 5: Verify
        print("\n" + "=" * 80)
        print("Step 5: Verification")
        print("=" * 80)
        verify_checkpoint(output_path)

        print("\n" + "=" * 80)
        print("Conversion Complete!")
        print("=" * 80)
        print(f"\nTo use this checkpoint, update your .env file:")
        print(f"  MODEL_CHECKPOINT_PATH={output_path}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

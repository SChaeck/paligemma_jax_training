#!/usr/bin/env python3
"""
Convert XVR-trained PaliGemma checkpoint to OpenPI format.

This script prepares the XVR-trained checkpoint for integration with OpenPI's
pi0.5 model and LIBERO benchmarking.

Usage:
    python scripts/convert_to_openpi.py \
        --input outputs/production/checkpoints/checkpoint_final.npz \
        --output ../openpi/xvr_trained_paligemma.npz
"""

import argparse
import numpy as np
from pathlib import Path


def convert_checkpoint(input_path: str, output_path: str, verify: bool = True):
    """
    Convert XVR-trained checkpoint to OpenPI-compatible format.
    
    The main change is ensuring the key format matches what OpenPI expects:
    - Input format: params/img/... and params/llm/...
    - Output format: same (OpenPI's PaliGemmaWeightLoader handles this)
    
    Args:
        input_path: Path to XVR-trained checkpoint (.npz)
        output_path: Path to save converted checkpoint
        verify: Whether to verify the conversion
    """
    print(f"Converting checkpoint for OpenPI...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    
    # Load input checkpoint
    print("\nLoading input checkpoint...")
    data = np.load(input_path, allow_pickle=False)
    keys = list(data.files)
    print(f"  Found {len(keys)} parameter arrays")
    
    # Verify format
    has_params_prefix = all(k.startswith("params/") for k in keys)
    if not has_params_prefix:
        print("  WARNING: Keys don't have 'params/' prefix. Adding...")
        converted = {f"params/{k}": data[k] for k in keys}
    else:
        print("  Keys already have 'params/' prefix. Good!")
        converted = {k: data[k] for k in keys}
    
    # Check for expected keys
    expected_prefixes = ["params/img/", "params/llm/"]
    for prefix in expected_prefixes:
        matching = [k for k in converted.keys() if k.startswith(prefix)]
        print(f"  {prefix}*: {len(matching)} keys")
    
    # Save converted checkpoint
    print(f"\nSaving to {output_path}...")
    np.savez(output_path, **converted)
    
    # Get file size
    output_size = Path(output_path).stat().st_size / 1e9
    print(f"  Saved ({output_size:.2f} GB)")
    
    # Verify
    if verify:
        print("\nVerifying conversion...")
        loaded = np.load(output_path, allow_pickle=False)
        assert len(loaded.files) == len(converted), "Key count mismatch!"
        for k in list(loaded.files)[:3]:
            assert np.allclose(loaded[k], converted[k]), f"Value mismatch for {k}!"
        print("  Verification passed!")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"""
Next steps:
1. Copy to OpenPI directory:
   cp {output_path} /path/to/openpi/xvr_trained_paligemma.npz

2. Add XVRTrainedPaliGemmaLoader to openpi/src/openpi/training/weight_loaders.py

3. Add training config to openpi/src/openpi/training/config.py

4. Run LIBERO fine-tuning:
   cd /path/to/openpi
   uv run scripts/train.py --config pi05_xvr_libero

5. Run LIBERO benchmark:
   SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_xvr_libero ..." \\
   docker compose -f examples/libero/compose.yml up --build
""")


def main():
    parser = argparse.ArgumentParser(
        description="Convert XVR-trained PaliGemma checkpoint to OpenPI format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to XVR-trained checkpoint (.npz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save converted checkpoint",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification step",
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    convert_checkpoint(
        str(input_path),
        str(output_path),
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()


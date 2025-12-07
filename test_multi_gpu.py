#!/usr/bin/env python3
"""
Quick test to verify multi-GPU detection and pmap functionality.
"""
import os
import sys

# Don't force single GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
import jax.numpy as jnp

print("=" * 80)
print("Multi-GPU Detection Test")
print("=" * 80)

# Check device count
num_devices = jax.device_count()
print(f"\nNumber of JAX devices: {num_devices}")

# List all devices
print("\nDevices:")
for i, device in enumerate(jax.devices()):
    print(f"  [{i}] {device.platform}: {device.device_kind}")

# Test pmap
if num_devices > 1:
    print(f"\n✓ Multi-GPU detected! pmap can be used.")

    # Simple pmap test
    @jax.pmap
    def add_one(x):
        return x + 1

    # Create replicated input
    x = jnp.arange(num_devices)
    x_replicated = jax.device_put_replicated(x, jax.devices())

    print(f"\nTesting pmap:")
    print(f"  Input shape: {x.shape}")
    print(f"  Input: {x}")

    result = add_one(x_replicated)
    print(f"  Output shape: {result.shape}")
    print(f"  Output: {result}")
    print("\n✓ pmap test successful!")

else:
    print(f"\n✗ Only {num_devices} GPU detected. pmap will not be used.")
    print("  To enable multi-GPU:")
    print("    1. Remove CUDA_VISIBLE_DEVICES='0' from training scripts")
    print("    2. Set USE_PMAP=true in .env file")

print("\n" + "=" * 80)

"""
Phase 0 - Environment check.

Purpose: before we write any quantization code, confirm that PyTorch can
see your GPU, that CUDA is working, and that you have enough VRAM headroom
for what we're about to do.

Run this with: python check_environment.py
"""

import sys

def check_python_version():
    major, minor = sys.version_info.major, sys.version_info.minor
    print(f"Python version: {major}.{minor}")
    if (major, minor) < (3, 10):
        print("  Warning: recommend Python 3.10+ for best library compatibility.")


def check_torch_and_cuda():
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it first (see requirements.txt).")
        return

    print(f"PyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        print("  CUDA is not visible to PyTorch. This usually means either:")
        print("  1) You installed the CPU-only build of torch, or")
        print("  2) Your NVIDIA drivers / CUDA toolkit aren't set up correctly.")
        return

    # If CUDA is available, inspect the actual device
    device_name = torch.cuda.get_device_name(0)
    total_mem_bytes = torch.cuda.get_device_properties(0).total_memory
    total_mem_gb = total_mem_bytes / (1024 ** 3)

    print(f"GPU detected: {device_name}")
    print(f"Total VRAM: {total_mem_gb:.2f} GB")

    if total_mem_gb < 7.5:
        print("  Note: less than 8GB VRAM detected. 4-bit quantized 8B models")
        print("  should still fit, but keep context length modest during testing.")
    else:
        print("  VRAM looks sufficient for 4-bit and 8-bit 8B model experiments.")


def check_key_libraries():
    # These are the libraries Phase 1-3 will depend on.
    # We check them now so a missing dependency doesn't surprise us mid-experiment.
    libs = ["transformers", "accelerate", "bitsandbytes", "datasets"]
    for lib in libs:
        try:
            module = __import__(lib)
            version = getattr(module, "__version__", "unknown")
            print(f"{lib}: installed (version {version})")
        except ImportError:
            print(f"{lib}: NOT installed")


if __name__ == "__main__":
    print("=== Phase 0 Environment Check ===\n")
    check_python_version()
    print()
    check_torch_and_cuda()
    print()
    check_key_libraries()
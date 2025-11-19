import os
import sys
import subprocess
from setuptools import setup, find_packages

# 1. Check for PyTorch manually.
# We do not list it in install_requires to prevent pip from uninstalling the user's
# existing specific version (e.g., torch+cu118) to install a generic one.
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    print("\n\033[91m[ERROR] PyTorch not found.\033[0m")
    print("This package requires PyTorch to be installed manually first.")
    print("Please install it via instructions at: https://pytorch.org/get-started/locally/")
    sys.exit(1)

def check_cuda_availability():
    if not torch.cuda.is_available():
        # We raise a warning instead of erroring out to allow building
        # in environments where GPU isn't currently active but CUDA tools are present
        print("WARNING: CUDA device not available. Building only is still possible if nvcc is found.")
        return False

    try:
        subprocess.run(['nvcc', '--version'], check=True, capture_output=True)
        return True
    except FileNotFoundError:
        print("ERROR: nvcc not found. Install CUDA toolkit and add to PATH.")
        sys.exit(1)

def get_cuda_arch_flags():
    """Get CUDA architecture flags for the detected GPU."""
    # If no GPU is detected, fall back to common architectures
    if not torch.cuda.is_available():
        print("Warning: No GPU detected. Building for common architectures (7.0, 7.5, 8.0, 8.6, 9.0)...")
        return [
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_90,code=sm_90",
        ]

    try:
        major, minor = torch.cuda.get_device_capability()
        # It's good practice to build for the detected GPU *and* forward compatible PTX
        arch = f"{major}{minor}"
        print(f"Detected CUDA capability {major}.{minor}")
        return [
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            f"-gencode=arch=compute_{arch},code=compute_{arch}" # Forward compatibility
        ]
    except Exception as e:
        print(f"Warning: Could not detect GPU architectures ({e})")
        return []

SKIP_CUDA_BUILD = os.environ.get("SJLT_SKIP_CUDA_BUILD", "0") == "1"

ext_modules = []
cmdclass = {}

if not SKIP_CUDA_BUILD:
    # Ensure CUDA is usable
    if check_cuda_availability() or torch.cuda.is_available() or os.environ.get("FORCE_CUDA_BUILD"):
        ext_modules = [
            CUDAExtension(
                name="sjlt._C",
                sources=["sjlt/kernels/sjlt_kernel.cu"],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": [
                        "-O3",
                        "--use_fast_math",
                        "-Xptxas=-v",
                        "--expt-relaxed-constexpr",
                    ] + get_cuda_arch_flags(),
                },
            )
        ]
        cmdclass = {"build_ext": BuildExtension}
    else:
        print("Skipping CUDA compilation (CUDA not found).")

setup(
    name="sjlt",
    version="0.1.6.1",
    author="Pingbang Hu",
    description="A PyTorch package for Sparse Johnson-Lindenstrauss Transform with CUDA.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    # IMPORTANT: Do NOT list 'torch' here.
    # It lets the user bring their own compatible version.
    install_requires=[
        # "torch",  <-- Removed
    ],
    python_requires=">=3.8",
    zip_safe=False,
)
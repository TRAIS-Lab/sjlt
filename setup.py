import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

def get_cuda_arch_flags():
    """
    Get CUDA architecture flags based on detected GPU capability.
    Falls back to common architectures if detection fails.
    """
    arch_flags = []

    if torch.cuda.is_available():
        try:
            # Get capability of the first GPU
            major, minor = torch.cuda.get_device_capability(0)

            # Generate architecture flags for detected GPU
            compute_capability = f"{major}{minor}"
            arch_flags = [
                f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
                f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}"  # For forward compatibility
            ]

            print(f"Detected CUDA capability: {major}.{minor}, using optimized compilation")

        except Exception as e:
            print(f"Could not detect CUDA capability ({e}), using default architectures")

    # If detection failed or no CUDA, use common architectures
    if not arch_flags:
        arch_flags = [
            "-gencode=arch=compute_70,code=sm_70",   # V100, T4
            "-gencode=arch=compute_75,code=sm_75",   # RTX 20 series, T4
            "-gencode=arch=compute_80,code=sm_80",   # A100, RTX 30 series
            "-gencode=arch=compute_86,code=sm_86",   # RTX 30 series
            "-gencode=arch=compute_89,code=sm_89",   # RTX 40 series
            "-gencode=arch=compute_90,code=sm_90",   # H100
        ]
        print("Using default CUDA architectures for broad compatibility")

    return arch_flags

def build_cuda_extension():
    """Build CUDA extension with error handling"""

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please install CUDA toolkit and PyTorch with CUDA support.\n"
            "Visit: https://pytorch.org/get-started/locally/"
        )

    # Check if nvcc is available
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError()
        print(f"Found CUDA compiler: nvcc")
    except FileNotFoundError:
        raise RuntimeError(
            "CUDA compiler (nvcc) not found. Please install CUDA toolkit.\n"
            "Visit: https://developer.nvidia.com/cuda-downloads"
        )

    # Build the extension
    return CUDAExtension(
        name="sjlt._C",
        sources=[
            "sjlt/kernels/sjlt_kernel.cu"
        ],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-std=c++14",
                "-fPIC"
            ],
            "nvcc": [
                "-O3",
                "--extended-lambda",
                "--expt-relaxed-constexpr",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--use_fast_math",
                "-Xptxas=-v",  # Verbose output for debugging
            ] + get_cuda_arch_flags()
        },
        include_dirs=[
            # Add any additional include directories here if needed
        ]
    )

# Conditional extension building
ext_modules = []
cmdclass = {}

try:
    ext_modules = [build_cuda_extension()]
    cmdclass = {"build_ext": BuildExtension}
    print("CUDA extension will be built")
except Exception as e:
    print(f"Warning: CUDA extension cannot be built: {e}")
    print("Package will be installed without CUDA support")

# Main setup
setup(
    name="sjlt",
    version="0.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,  # Required for CUDA extensions
)
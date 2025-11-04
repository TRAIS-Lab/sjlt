import os
import sys
import subprocess
from setuptools import setup, find_packages


def check_cuda_availability():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Install PyTorch with CUDA support.")
    try:
        subprocess.run(['nvcc', '--version'], check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError("nvcc not found. Install CUDA toolkit and add to PATH.")

def get_cuda_arch_flags():
    """Get CUDA architecture flags for the detected GPU."""
    import torch
    if not torch.cuda.is_available():
        return []

    try:
        major, minor = torch.cuda.get_device_capability()
        arch_flag = f"--generate-code=arch=compute_{major}{minor},code=sm_{major}{minor}"
        print(f"Detected CUDA capability {major}.{minor}")
        return [arch_flag]
    except Exception as e:
        print(f"Warning: Could not detect GPU, using default architectures ({e})")
        return [
            "--generate-code=arch=compute_70,code=sm_70",
            "--generate-code=arch=compute_75,code=sm_75",
            "--generate-code=arch=compute_86,code=sm_86",
            "--generate-code=arch=compute_90,code=sm_90",
        ]


SKIP_CUDA_BUILD = os.environ.get("SJLT_SKIP_CUDA_BUILD", "0") == "1"

if SKIP_CUDA_BUILD:
    print("WARNING: Skipping CUDA build. Package will not work at runtime!")
    ext_modules = []
    cmdclass = {}
else:
    # Import torch here so it's only required when actually building
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError:
        print("ERROR: PyTorch not found. Please install PyTorch with CUDA support first.")
        print("Visit https://pytorch.org/get-started/locally/ for installation instructions.")
        sys.exit(1)

    try:
        check_cuda_availability()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

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


setup(
    name="sjlt",
    version="0.1.5.1",
    author="Pingbang Hu",
    description="A PyTorch package for Sparse Johnson-Lindenstrauss Transform with CUDA.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        "torch",
    ],
    python_requires=">=3.8",
    zip_safe=False,
)
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# A helper function to make sure CUDA is available and nvcc is found
def check_cuda_availability():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please install PyTorch with CUDA support.\n"
            "Visit: https://pytorch.org/get-started/locally/"
        )
    try:
        import subprocess
        subprocess.run(['nvcc', '--version'], check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "CUDA compiler (nvcc) not found. Please ensure the CUDA toolkit is installed and in your PATH."
        )

def get_cuda_arch_flags():
    """
    Get CUDA architecture flags for the detected GPU.
    Falls back to common architectures if detection fails.
    """
    if not torch.cuda.is_available():
        return []

    try:
        major, _ = torch.cuda.get_device_capability()
        arch_flag = f"--generate-code=arch=compute_{major}{_},code=sm_{major}{_}"
        print(f"Detected CUDA capability {major}.{_}, using arch flag: {arch_flag}")
        return [arch_flag]
    except Exception as e:
        print(f"Warning: Could not detect CUDA capability, falling back to default architectures. Error: {e}")
        # Common architectures for broad compatibility
        return [
            "--generate-code=arch=compute_70,code=sm_70",  # V100
            "--generate-code=arch=compute_75,code=sm_75",  # T4, RTX 20-series
            "--generate-code=arch=compute_86,code=sm_86",  # RTX 30-series, A100
            "--generate-code=arch=compute_90,code=sm_90",  # H100
        ]

# Build the CUDA extension
try:
    check_cuda_availability()
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
except RuntimeError as e:
    print(f"Skipping CUDA extension build due to: {e}")
    ext_modules = []
    cmdclass = {}

# Main setup
setup(
    name="sjlt",
    version="0.1.5",
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
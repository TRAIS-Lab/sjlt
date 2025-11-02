"""
sjlt - Sparse Johnson-Lindenstrauss Transform with CUDA acceleration

A PyTorch extension that provides GPU-accelerated sparse random projections
using the Johnson-Lindenstrauss lemma for dimensionality reduction.
"""

__version__ = "0.1.5"
__author__ = "Pingbang Hu"

import torch
import warnings

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# Try to import the compiled extension
EXTENSION_AVAILABLE = False
if CUDA_AVAILABLE:
    try:
        from ._C import sjlt_projection_cuda
        EXTENSION_AVAILABLE = True
    except ImportError as e:
        warnings.warn(
            f"sjlt CUDA extension could not be loaded: {e}\n"
            "Make sure the package was installed correctly with CUDA support.\n"
            "Try: pip install sjlt --no-cache-dir --force-reinstall"
        )
else:
    warnings.warn(
        "CUDA is not available. sjlt CUDA functionality will not work.\n"
        "Please install PyTorch with CUDA support."
    )

# Import main classes
from .core import SJLTProjection

# Define what gets imported with "from sjlt import *"
__all__ = [
    "SJLTProjection",
    "CUDA_AVAILABLE",
    "EXTENSION_AVAILABLE",
    "__version__"
]

def get_cuda_info():
    """Get information about CUDA setup"""
    info = {
        "cuda_available": CUDA_AVAILABLE,
        "extension_available": EXTENSION_AVAILABLE,
    }

    if CUDA_AVAILABLE:
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device() if torch.cuda.is_initialized() else None,
        })

        # Get device names
        if torch.cuda.device_count() > 0:
            info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    return info
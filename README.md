# sjlt

Sparse Johnson-Lindenstrauss Transform with CUDA acceleration for PyTorch.

## Features

- GPU-accelerated sparse random projections
- Supports float32, float64, and bfloat16 data types
- Optimized CUDA kernels for high performance
- Easy integration with PyTorch workflows

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0 with CUDA support
- CUDA Toolkit (version compatible with your PyTorch installation)
- C++ compiler (GCC 7-11 recommended)

### Install from PyPI

```bash
pip install sjlt
```

### Install from Source

```bash
git clone https://github.com/TRAIS-Lab/sjlt
cd sjlt
pip install -e .
```

## Quick Start

Our sjlt implementation has the following parameters:

- `original_dim`: input dimension
- `proj_dim`: output dimension
- `c`: sparsity parameter, i.e., non-zeros per column (default: `1`)
- `threads`: CUDA threads per block (default: `1024`)
- `fixed_blocks`: CUDA blocks to use (default: `84`)


> We note that the input is supposed to have `batch_dim`, i.e., `input.shape()` should be `(batch_size, original_dim)` and `output.shape()` will be `(batch_size, proj_dim)`.

The following is a simple snippet of using our SJLT CUDA kernel:

```python
import torch
from sjlt import SJLTProjection

# Create projection: 1024 -> 128 dimensions with sparsity 4
proj = SJLTProjection(original_dim=1024, proj_dim=128, c=4)

# Project some data
x = torch.randn(100, 1024, device='cuda')
y = proj(x)  # Shape: [100, 128]

print(f"Compression ratio: {proj.get_compression_ratio():.2f}x")
print(f"Sparsity: {proj.get_sparsity_ratio():.1%}")
```

## Troubleshooting

If installation fails:

1. Ensure CUDA toolkit is installed and `nvcc` is in `PATH`
2. Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`
3. Try reinstalling: `pip install sjlt --no-cache-dir --force-reinstall`

# Sparse Johnson-Lindenstrauss Transform CUDA Kernel

This is a simple repository for *Sparse Johnson-Lindenstrauss Transform* with CUDA acceleration for PyTorch.

## Features

- GPU-accelerated sparse random projections
- Supports float16, float32, float64, and bfloat16 data types
- Optimized CUDA kernels for high performance
- Easy integration with PyTorch workflows

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0 with CUDA support
- CUDA Toolkit (version compatible with your PyTorch installation)
- C++ compiler (GCC 7-11 recommended)

### Install from PyPI

To build the CUDA SJLT CUDA kernel, you will need to make sure that `nvcc -V` and `torch.version.cuda` gives the same CUDA version. Then, you can install `sjlt` via:

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

Our SJLT implementation accepts the following parameters:

- `original_dim`: input dimension
- `proj_dim`: output dimension
- `c`: sparsity parameter, i.e., non-zeros per column (default: `1`)
- `threads`: CUDA threads per block (default: `1024`)
- `fixed_blocks`: CUDA blocks to use (default: `84`)


> [!Note]
> The input is supposed to have `batch_dim`, i.e., `input.shape()` should be `(batch_size, original_dim)` and `output.shape()` will be `(batch_size, proj_dim)`.

The following is a simple snippet of using our SJLT CUDA kernel:

```python
import torch
from sjlt import SJLTProjection

# Create projection: 1024 -> 128 dimensions with sparsity 4
proj = SJLTProjection(original_dim=1024, proj_dim=128, c=4)

# Project some data
x = torch.randn(100, 1024, device='cuda')
y = proj(x)  # Shape: [100, 128]
```

## Profile Example

To profile the performance of the SJLT CUDA kernel, you can use the provided [profile](https://github.com/TRAIS-Lab/sjlt/blob/main/test/profile.ipynb) notebook. This benchmarks the projection speed for different input sizes and sparsity levels.

<!-- Image -->
![SJLT Example](Figures/profile.png)

## Troubleshooting

Installing a CUDA kernel can be tricky. Here, I have gathered several common errors I have encountered, with their corresponding solutions.

1. `pip install sjlt` doesn't work: In this case, please try to build it in your environment. This means to try `pip install sjlt --no-cache-dir --force-reinstall`
2. CUDA version mismatch: Please make sure that you have a compatible CUDA version of `nvcc` as well as the CUDA version that is used to build PyTorch. If this is the case, but you still encounter the following errors:
   ```bash
   RuntimeError:
   The detected CUDA version (11.8) mismatches the version that was used to compile
   PyTorch (12.6). Please make sure to use the same CUDA versions.
   ```
   This is because the default [isolation building behavior](https://github.com/vllm-project/vllm/issues/1453#issuecomment-1951453221) of `pip install` (this applies to both "Install from PyPI" and "Install from Source"), even if `nvcc -V` and `torch.version.cuda` give the same CUDA version. In this case, using `--no-build-isolation` to force pip to build using your current virtual environment. With 1., please try `pip install sjlt --no-build-isolation --no-cache-dir --force-reinstall`
3. Unsupported GNU version: With 1. and 2., if you then encounter the following:
   ```bash
   error -- unsupported GNU version! gcc versions later than 11 are not supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.
   ```
   This means your GCC version is too new for your CUDA version. Please ensure that you have the correct GCC version installed, and once you have done so, you can use the corresponding GCC version by setting the environment variables such as `CC=gcc-11 CXX=g++-11 pip install sjlt --no-build-isolation --no-cache-dir --force-reinstall` in the case of `CUDA=11.8`.

In summary:

1. Ensure CUDA toolkit is installed and `nvcc` is in `PATH`
2. Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`
3. Try clean reinstalling: `pip install sjlt --no-build-isolation --no-cache-dir --force-reinstall`
4. If GCC version is too new, install compatible version and set `CC` and `CXX` environment variables accordingly.

## Reference

1. [A Sparse Johnson-Lindenstrauss Transform](https://arxiv.org/abs/1004.4240)
2. [Sparser Johnson-Lindenstrauss Transforms](https://arxiv.org/abs/1012.1577)
3. [GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection](https://arxiv.org/abs/2505.18976)

## Citation

If you find this repository valuable, please give it a star! Got any questions or feedback? Feel free to open an issue. Using this in your work? Please reference us using the provided citation:

```bibtex
@inproceedings{hu2025grass,
  author    = {Pingbang Hu and Joseph Melkonian and Weijing Tang and Han Zhao and Jiaqi W. Ma},
  title     = {GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

> As this repository is an effort from the [GraSS](https://github.com/TRAIS-Lab/GraSS) project.
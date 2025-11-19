# Contributing to SJLT

Thank you for your interest in contributing to SJLT! This guide will help you set up your development environment, build the project, and publish updates to PyPI.

## Table of Contents

- [Development Setup](#development-setup)
- [Building from Source](#building-from-source)
- [Running Tests](#running-tests)
- [Publishing to PyPI](#publishing-to-pypi)

## Development Setup

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.9.0 with CUDA support
- CUDA Toolkit (compatible with your PyTorch installation)
- C++ compiler (GCC 7-11 recommended for CUDA 11.8)
- Git

### Recommended Environment Setup

We recommend using conda for environment management:

```bash
# Create a new conda environment
conda create -n sjlt-dev python=3.10
conda activate sjlt-dev

# Install CUDA toolkit (adjust version as needed)
conda install -c "nvidia/label/cuda-11.8.0" cudatoolkit

# Install PyTorch with matching CUDA version
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Clone the repository
git clone https://github.com/TRAIS-Lab/sjlt
cd sjlt

# Install in editable mode
pip install -e .
```

### Verify Installation

Check that CUDA and PyTorch are properly configured:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'PyTorch CUDA version: {torch.version.cuda}')"

# Check nvcc version
nvcc -V

# Test the package
python -c "from sjlt import SJLTProjection; print('SJLT imported successfully!')"
```

## Building from Source

### Standard Build

```bash
# Clean previous builds
rm -rf dist/ build/ sjlt.egg-info/

# Build the package
pip install -e .
```

### Build Options

**Skip CUDA Build** (for CPU-only development):
```bash
export SJLT_SKIP_CUDA_BUILD=1
pip install -e .
```

**Force CUDA Build** (even if GPU not detected):
```bash
export FORCE_CUDA_BUILD=1
pip install -e .
```

**Build without isolation** (use current environment):
```bash
pip install -e . --no-build-isolation
```

### Troubleshooting Build Issues

**CUDA Version Mismatch:**
```bash
# Ensure nvcc and PyTorch use the same CUDA version
nvcc -V
python -c "import torch; print(torch.version.cuda)"

# If mismatch, reinstall with --no-build-isolation
pip install -e . --no-build-isolation --no-cache-dir --force-reinstall
```

**Unsupported GCC Version:**
```bash
# Install compatible GCC version (e.g., GCC 11 for CUDA 11.8)
# Then specify the compiler
CC=gcc-11 CXX=g++-11 pip install -e . --no-build-isolation --no-cache-dir --force-reinstall
```

## Running Tests

### Run Basic Tests

```bash
# Navigate to test directory
cd test

# Run profile notebook (requires Jupyter)
jupyter notebook profile.ipynb
```

### Manual Testing

```python
import torch
from sjlt import SJLTProjection

# Test basic projection
proj = SJLTProjection(original_dim=1024, proj_dim=128, c=4)
x = torch.randn(100, 1024, device='cuda')
y = proj(x)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")

# Test different data types
for dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
    x = torch.randn(10, 1024, device='cuda', dtype=dtype)
    y = proj(x)
    print(f"dtype {dtype}: ✓")
```

## Publishing to PyPI

### Prerequisites

1. **Create PyPI Account**
   - Register at https://pypi.org/account/register/
   - Verify your email
   - (Optional) Register at https://test.pypi.org/account/register/ for testing

2. **Install Publishing Tools**
   ```bash
   pip install --upgrade build twine
   ```

3. **Configure API Token** (recommended for security)
   - Go to PyPI Account Settings → API tokens
   - Create a new token with upload permissions
   - Create `~/.pypirc`:
     ```ini
     [pypi]
     username = __token__
     password = pypi-YOUR-API-TOKEN-HERE

     [testpypi]
     username = __token__
     password = pypi-YOUR-TESTPYPI-TOKEN-HERE
     ```

### Publishing Workflow

#### Step 1: Update Version

Update version in **both** files:
- `pyproject.toml` (line 7): `version = "<new_version>"`
- `setup.py` (line 87): `version="<new_version>"`

#### Step 2: Clean Previous Builds

```bash
rm -rf dist/ build/ sjlt.egg-info/
```

#### Step 3: Build Distribution Packages

```bash
# Build only source distribution (recommended for CUDA packages)
python -m build --sdist --no-isolation
```

This creates:
- `dist/sjlt-X.X.X.tar.gz` (source distribution)

**Note:** We only distribute the source distribution (sdist) for this package because:
- Users need to compile with their specific CUDA version
- Different PyTorch installations require different compilation
- PyPI doesn't accept generic `linux_x86_64` wheels (requires `manylinux` tags)
- CUDA packages are best built in the user's environment

#### Step 4: Validate Distribution

```bash
# Check package metadata and description
twine check dist/*
```

Fix any errors or warnings before proceeding.

#### Step 5: Test on TestPyPI (Recommended)

```bash
# Upload to TestPyPI (only source distribution)
twine upload --repository testpypi dist/*.tar.gz

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --no-deps sjlt

# Test that it works
python -c "from sjlt import SJLTProjection; print('Success!')"
```

#### Step 6: Upload to PyPI

```bash
# Upload to production PyPI (only source distribution)
twine upload dist/*.tar.gz
```

#### Step 7: Verify

- Visit https://pypi.org/project/sjlt/
- Install and test: `pip install sjlt`

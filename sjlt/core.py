"""
Core sjlt implementation with CUDA acceleration.
"""

import torch
import torch.nn as nn
import math
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

class SJLTProjection(nn.Module):
    """
    Sparse Johnson-Lindenstrauss Transform implemented with CUDA kernels.

    This module provides GPU-accelerated sparse random projections for
    dimensionality reduction while approximately preserving distances.

    Args:
        original_dim (int): Original dimension of input vectors
        proj_dim (int): Target projection dimension
        c (int): Number of non-zeros per column (sparsity parameter)
        threads (int): Number of CUDA threads per block (must be multiple of 32)
        fixed_blocks (int): Number of CUDA blocks to use
        device (str or torch.device): Device to run computation on

    Example:
        >>> proj = SJLTProjection(original_dim=1024, proj_dim=128, c=4)
        >>> x = torch.randn(100, 1024, device='cuda')
        >>> y = proj(x)  # Shape: [100, 128]
    """

    def __init__(
        self,
        original_dim: int,
        proj_dim: int,
        c: int = 1,
        threads: int = 1024,
        fixed_blocks: int = 84,
        device: Union[str, torch.device] = 'cuda'
    ):
        super(SJLTProjection, self).__init__()

        # Import check
        from . import EXTENSION_AVAILABLE
        if not EXTENSION_AVAILABLE:
            raise RuntimeError(
                "sjlt CUDA extension is not available. "
                "Please ensure CUDA is installed and the package was compiled correctly."
            )

        # Validate arguments
        if original_dim <= 0 or proj_dim <= 0:
            raise ValueError("Dimensions must be positive")
        if c <= 0:
            raise ValueError("Sparsity parameter c must be positive")
        if proj_dim < c:
            raise ValueError("Projection dimension must be >= sparsity parameter c")

        self.original_dim = original_dim
        self.proj_dim = proj_dim
        self.c = c

        # Ensure threads is a multiple of 32 (warp size) for optimal performance
        self.threads = max(32, (threads // 32) * 32)
        self.fixed_blocks = fixed_blocks
        self.device = torch.device(device)

        # Generate sparse random matrix components
        # Each input dimension maps to c random output dimensions with random signs
        self.register_buffer(
            'rand_indices',
            torch.randint(0, proj_dim, (original_dim, c), device=self.device, dtype=torch.long)
        )
        self.register_buffer(
            'rand_signs',
            (torch.randint(0, 2, (original_dim, c), device=self.device) * 2 - 1).to(torch.int8)
        )

        # Pre-compute normalization factor
        self.normalization_factor = 1.0 / math.sqrt(c)

        logger.debug(
            f"Initialized SJLT projection: {original_dim} -> {proj_dim}, "
            f"sparsity={c}, device={self.device}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SJLT projection to input tensor.

        Args:
            x: Input tensor of shape [batch_size, original_dim]

        Returns:
            Projected tensor of shape [batch_size, proj_dim]
        """
        from ._C import sjlt_projection_cuda

        # Input validation
        if x.dim() != 2:
            raise ValueError(f"Input must be 2D tensor, got {x.dim()}D")
        if x.size(1) != self.original_dim:
            raise ValueError(
                f"Input dimension {x.size(1)} doesn't match expected {self.original_dim}"
            )

        # Move input to correct device if needed
        if x.device != self.device:
            x = x.to(self.device)

        # Ensure indices and signs are on same device (safety check)
        rand_indices = self.rand_indices.to(x.device)
        rand_signs = self.rand_signs.to(x.device)

        # Apply SJLT projection using CUDA kernel
        try:
            output = sjlt_projection_cuda(
                x,
                rand_indices,
                rand_signs,
                self.proj_dim,
                self.c,
                self.threads,
                self.fixed_blocks
            )[0]

            return output

        except Exception as e:
            raise RuntimeError(f"CUDA kernel execution failed: {e}")

    def extra_repr(self) -> str:
        """Extra information for string representation"""
        return (
            f'original_dim={self.original_dim}, proj_dim={self.proj_dim}, '
            f'c={self.c}, device={self.device}'
        )

    def get_compression_ratio(self) -> float:
        """Get the compression ratio of this projection"""
        return self.original_dim / self.proj_dim

    def get_sparsity_ratio(self) -> float:
        """Get the sparsity ratio of the projection matrix"""
        total_elements = self.original_dim * self.proj_dim
        nonzero_elements = self.original_dim * self.c
        return 1.0 - (nonzero_elements / total_elements)

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the transpose of the SJLT projection matrix to input tensor.

        This operation computes S^T @ y, where S is the sparse projection matrix.
        It maps from the projected space back to the original dimension space.

        Args:
            y: Input tensor of shape [batch_size, proj_dim]

        Returns:
            Transposed projection of shape [batch_size, original_dim]

        Example:
            >>> proj = SJLTProjection(original_dim=1024, proj_dim=128, c=4)
            >>> x = torch.randn(100, 1024, device='cuda')
            >>> y = proj(x)  # Forward projection: [100, 1024] -> [100, 128]
            >>> x_reconstructed = proj.transpose(y)  # Transpose: [100, 128] -> [100, 1024]
        """
        from ._C import sjlt_transpose_cuda

        # Input validation
        if y.dim() != 2:
            raise ValueError(f"Input must be 2D tensor, got {y.dim()}D")
        if y.size(1) != self.proj_dim:
            raise ValueError(
                f"Input dimension {y.size(1)} doesn't match expected {self.proj_dim}"
            )

        # Move input to correct device if needed
        if y.device != self.device:
            y = y.to(self.device)

        # Ensure indices and signs are on same device (safety check)
        rand_indices = self.rand_indices.to(y.device)
        rand_signs = self.rand_signs.to(y.device)

        # Apply SJLT transpose using CUDA kernel
        try:
            output = sjlt_transpose_cuda(
                y,
                rand_indices,
                rand_signs,
                self.original_dim,
                self.c,
                self.threads,
                self.fixed_blocks
            )[0]

            return output

        except Exception as e:
            raise RuntimeError(f"CUDA kernel execution failed: {e}")
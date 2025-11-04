import pytest
import torch
import sys
import os

# Add the parent directory to the path to import sjlt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import sjlt
    SJLT_AVAILABLE = sjlt.EXTENSION_AVAILABLE
except ImportError:
    SJLT_AVAILABLE = False

@pytest.mark.skipif(not SJLT_AVAILABLE, reason="sjlt CUDA extension not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSJLTProjection:

    def test_basic_functionality(self):
        """Test basic sjlt projection"""
        proj = sjlt.SJLTProjection(original_dim=100, proj_dim=50, c=4)

        # Test with random input
        x = torch.randn(10, 100, device='cuda')
        y = proj(x)

        assert y.shape == (10, 50)
        assert y.device == x.device

    def test_different_batch_sizes(self):
        """Test with different batch sizes"""
        proj = sjlt.SJLTProjection(original_dim=64, proj_dim=32, c=2)

        for batch_size in [1, 5, 100, 1000]:
            x = torch.randn(batch_size, 64, device='cuda')
            y = proj(x)
            assert y.shape == (batch_size, 32)

    def test_different_dtypes(self):
        """Test with different data types"""
        proj = sjlt.SJLTProjection(original_dim=32, proj_dim=16, c=3)

        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            x = torch.randn(5, 32, device='cuda', dtype=dtype)
            y = proj(x)
            assert y.dtype == dtype
            assert y.shape == (5, 16)

    def test_error_handling(self):
        """Test error conditions"""
        # Test invalid dimensions
        with pytest.raises(ValueError):
            sjlt.SJLTProjection(original_dim=0, proj_dim=10)

        with pytest.raises(ValueError):
            sjlt.SJLTProjection(original_dim=10, proj_dim=0)

        # Test invalid sparsity
        with pytest.raises(ValueError):
            sjlt.SJLTProjection(original_dim=10, proj_dim=5, c=0)

        # Test dimension mismatch
        proj = sjlt.SJLTProjection(original_dim=10, proj_dim=5, c=2)
        x = torch.randn(5, 20, device='cuda')  # Wrong input dimension

        with pytest.raises(ValueError):
            proj(x)

    def test_compression_metrics(self):
        """Test compression and sparsity metrics"""
        proj = sjlt.SJLTProjection(original_dim=100, proj_dim=25, c=4)

        assert proj.get_compression_ratio() == 4.0
        expected_sparsity = 1.0 - (100 * 4) / (100 * 25)
        assert abs(proj.get_sparsity_ratio() - expected_sparsity) < 1e-6

    def test_transpose_basic(self):
        """Test basic transpose functionality"""
        proj = sjlt.SJLTProjection(original_dim=100, proj_dim=50, c=4)

        # Forward projection
        x = torch.randn(10, 100, device='cuda')
        y = proj(x)

        # Transpose projection
        x_back = proj.transpose(y)

        assert x_back.shape == (10, 100)
        assert x_back.device == x.device

    def test_transpose_dimensions(self):
        """Test transpose with different dimensions"""
        original_dim, proj_dim = 256, 64
        proj = sjlt.SJLTProjection(original_dim=original_dim, proj_dim=proj_dim, c=3)

        for batch_size in [1, 5, 100]:
            y = torch.randn(batch_size, proj_dim, device='cuda')
            x = proj.transpose(y)
            assert x.shape == (batch_size, original_dim)

    def test_transpose_dtypes(self):
        """Test transpose with different data types"""
        proj = sjlt.SJLTProjection(original_dim=64, proj_dim=32, c=2)

        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            y = torch.randn(5, 32, device='cuda', dtype=dtype)
            x = proj.transpose(y)
            assert x.dtype == dtype
            assert x.shape == (5, 64)

    def test_transpose_consistency(self):
        """Test that S^T applied after S maintains mathematical consistency"""
        proj = sjlt.SJLTProjection(original_dim=128, proj_dim=64, c=4)

        # Create input
        x = torch.randn(10, 128, device='cuda')

        # Apply forward then transpose
        y = proj(x)
        x_reconstructed = proj.transpose(y)

        # The reconstruction won't be exact (SJLT is not invertible),
        # but should maintain the same shape and be a valid operation
        assert x_reconstructed.shape == x.shape
        assert not torch.isnan(x_reconstructed).any()
        assert not torch.isinf(x_reconstructed).any()

    def test_transpose_error_handling(self):
        """Test transpose error conditions"""
        proj = sjlt.SJLTProjection(original_dim=100, proj_dim=50, c=3)

        # Test wrong dimension
        y_wrong = torch.randn(5, 30, device='cuda')  # Should be 50, not 30
        with pytest.raises(ValueError):
            proj.transpose(y_wrong)

        # Test wrong number of dimensions
        y_1d = torch.randn(50, device='cuda')
        with pytest.raises(ValueError):
            proj.transpose(y_1d)

    def test_transpose_linearity(self):
        """Test that transpose operation is linear: S^T(ay + bz) = a*S^T(y) + b*S^T(z)"""
        proj = sjlt.SJLTProjection(original_dim=64, proj_dim=32, c=4)

        y1 = torch.randn(5, 32, device='cuda')
        y2 = torch.randn(5, 32, device='cuda')
        a, b = 2.5, -1.3

        # Compute S^T(a*y1 + b*y2)
        combined = proj.transpose(a * y1 + b * y2)

        # Compute a*S^T(y1) + b*S^T(y2)
        separate = a * proj.transpose(y1) + b * proj.transpose(y2)

        # Should be equal (within floating point precision)
        assert torch.allclose(combined, separate, rtol=1e-5, atol=1e-6)

def test_cuda_info():
    """Test CUDA info function"""
    info = sjlt.get_cuda_info()

    assert isinstance(info, dict)
    assert 'cuda_available' in info
    assert 'extension_available' in info

    if torch.cuda.is_available():
        assert 'cuda_version' in info
        assert 'device_count' in info

if __name__ == "__main__":
    pytest.main([__file__])
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
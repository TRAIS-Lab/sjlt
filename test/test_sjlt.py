import pytest
import torch
import sjlt

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

    def test_transpose_adjoint_property(self):
        """Test that S and S^T are adjoint: <Sx, y> = <x, S^T y>"""
        proj = sjlt.SJLTProjection(original_dim=128, proj_dim=64, c=4)

        # Create random vectors
        x = torch.randn(10, 128, device='cuda')
        y = torch.randn(10, 64, device='cuda')

        # Compute <Sx, y>
        Sx = proj(x)
        inner_product_1 = (Sx * y).sum(dim=1)

        # Compute <x, S^T y>
        STy = proj.transpose(y)
        inner_product_2 = (x * STy).sum(dim=1)

        # These should be equal (within numerical precision)
        # Note: SJLT uses sparse random matrices, so this should hold exactly
        assert torch.allclose(inner_product_1, inner_product_2, rtol=1e-4, atol=1e-5)

    def test_projection_idempotence_approximation(self):
        """Test that S^T S applied to a vector preserves it approximately

        For an orthogonal projection P, we have P^T P = I in the range of P.
        SJLT is not exactly orthogonal but should approximately preserve vectors.
        """
        proj = sjlt.SJLTProjection(original_dim=256, proj_dim=128, c=4)

        # Create random vectors
        x = torch.randn(20, 256, device='cuda')

        # Apply S then S^T
        y = proj(x)
        x_reconstructed = proj.transpose(y)

        # Check that the reconstruction has reasonable correlation with original
        # Normalize vectors for comparison
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        x_rec_norm = x_reconstructed / (x_reconstructed.norm(dim=1, keepdim=True) + 1e-8)

        # Compute cosine similarity
        cosine_sim = (x_norm * x_rec_norm).sum(dim=1)

        # The cosine similarity should be reasonably high (not perfect due to projection)
        # but should show that the direction is somewhat preserved
        mean_cosine_sim = cosine_sim.mean().item()

        # This is a sanity check - the reconstruction shouldn't be random
        assert mean_cosine_sim > 0.5, f"Mean cosine similarity too low: {mean_cosine_sim}"

    def test_transpose_gram_matrix_property(self):
        """Test that S^T S approximates a scaled identity matrix

        For random projections with JL properties, S^T S should be close to
        (original_dim / proj_dim) * I, at least in expectation.
        """
        proj = sjlt.SJLTProjection(original_dim=128, proj_dim=64, c=4)

        # Create a set of orthonormal vectors (identity matrix rows)
        # This makes it easier to analyze S^T S
        num_samples = 64
        I = torch.eye(num_samples, 128, device='cuda')

        # Apply S then S^T to identity
        SI = proj(I)
        STSI = proj.transpose(SI)

        # Extract the diagonal and off-diagonal elements
        diag_elements = torch.diagonal(STSI[:num_samples, :num_samples], 0)

        # The diagonal elements should be positive and relatively consistent
        assert (diag_elements > 0).all(), "Diagonal elements should be positive"

        # Check that diagonal variance is not too large (they should be similar)
        diag_mean = diag_elements.mean()
        diag_std = diag_elements.std()

        # Standard deviation should be much smaller than mean
        assert diag_std / diag_mean < 1.0, f"Diagonal elements too variable: mean={diag_mean}, std={diag_std}"

    def test_transpose_preserves_inner_products_approximately(self):
        """Test that inner products are approximately preserved through S^T S

        This is a key property of JL transforms: <x, y> ≈ <Sx, Sy> / scaling
        which implies <x, y> ≈ <x, S^T S y> with appropriate scaling
        """
        # Use a reasonable projection ratio
        proj = sjlt.SJLTProjection(original_dim=256, proj_dim=128, c=4)

        # Create pairs of random vectors
        num_pairs = 50
        x = torch.randn(num_pairs, 256, device='cuda')
        y = torch.randn(num_pairs, 256, device='cuda')

        # Compute original inner products
        original_inner = (x * y).sum(dim=1)

        # Compute inner products in projected space
        Sx = proj(x)
        Sy = proj(y)
        projected_inner = (Sx * Sy).sum(dim=1)

        # The projected inner product should correlate with the original
        # (they won't be exactly equal due to the random projection)
        correlation = torch.corrcoef(torch.stack([original_inner, projected_inner]))[0, 1]

        # Correlation should be positive and significant for a good random projection
        # Note: SJLT is sparse (c=4), so correlation won't be as high as dense projections
        assert correlation > 0.3, f"Inner product correlation too low: {correlation}"

        # Additionally, test via S^T S
        STSy = proj.transpose(proj(y))
        inner_via_STS = (x * STSy).sum(dim=1)

        # This should also correlate well with original
        correlation_STS = torch.corrcoef(torch.stack([original_inner, inner_via_STS]))[0, 1]
        assert correlation_STS > 0.5, f"S^T S inner product correlation too low: {correlation_STS}"

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
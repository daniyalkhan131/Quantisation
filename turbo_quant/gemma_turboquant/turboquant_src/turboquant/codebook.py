"""
Lloyd-Max optimal scalar quantizer for the Beta distribution arising from
random rotation of unit vectors on S^(d-1).

After random rotation, each coordinate follows:
    f(x) = C * (1 - x^2)^((d-3)/2)  on [-1, 1]

For d=128 this is very close to N(0, 1/128).

We solve the continuous k-means (Lloyd-Max) problem to find optimal centroids
and boundaries for a given bit-width b (2^b quantization levels).
"""

import numpy as np
from scipy import integrate
from scipy.special import gammaln
import torch

# Precomputed codebooks keyed by (dim, bits)
_CODEBOOK_CACHE = {}


def _beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """Probability density for a coordinate of a uniformly random unit vector in R^d.

    f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
    """
    if np.any(np.abs(x) >= 1.0):
        result = np.zeros_like(x, dtype=float)
        mask = np.abs(x) < 1.0
        if np.any(mask):
            log_norm = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
            result[mask] = np.exp(log_norm + ((d - 3) / 2) * np.log(1 - x[mask] ** 2))
        return result
    log_norm = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
    return np.exp(log_norm + ((d - 3) / 2) * np.log(1 - x**2))


def _integrate(f, a: float, b: float) -> float:
    """Numerically integrate f from a to b using scipy.integrate.quad."""
    result, _ = integrate.quad(f, a, b, limit=100)
    return result


def compute_lloyd_max_codebook(
    d: int, bits: int, max_iter: int = 1000, tol: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Compute optimal Lloyd-Max centroids and boundaries for the Beta distribution.

    Args:
        d: Dimension of the vectors (determines the Beta distribution shape).
        bits: Number of bits per coordinate (2^bits quantization levels).
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance on centroid change.

    Returns:
        (centroids, boundaries) where:
            centroids: array of 2^bits values in [-1, 1]
            boundaries: array of 2^bits - 1 values (midpoints between centroids)
    """
    n_levels = 2**bits
    pdf = lambda x: _beta_pdf(np.atleast_1d(np.array(x, dtype=float)), d).item()

    # Initialize centroids uniformly in the support region
    # For d=128, most mass is in [-0.3, 0.3], but we span [-1, 1]
    centroids = np.linspace(-0.99, 0.99, n_levels)

    for iteration in range(max_iter):
        # E-step: boundaries are midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # M-step: update centroids as conditional means
        # Full boundaries: -1, b1, b2, ..., b_{n-1}, 1
        full_bounds = np.concatenate([[-1.0], boundaries, [1.0]])
        new_centroids = np.zeros(n_levels)

        for i in range(n_levels):
            lo, hi = full_bounds[i], full_bounds[i + 1]
            mass = _integrate(pdf, lo, hi)
            if mass > 1e-15:
                mean = _integrate(lambda x: x * pdf(x), lo, hi)
                new_centroids[i] = mean / mass
            else:
                # Keep old centroid if interval has negligible mass
                new_centroids[i] = centroids[i]

        # Check convergence
        delta = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if delta < tol:
            break

    # Final boundaries
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids, boundaries


def compute_distortion(d: int, bits: int, centroids: np.ndarray, boundaries: np.ndarray) -> float:
    """Compute per-coordinate MSE distortion for the given codebook."""
    pdf = lambda x: _beta_pdf(np.atleast_1d(np.array(x, dtype=float)), d).item()
    full_bounds = np.concatenate([[-1.0], boundaries, [1.0]])

    total_mse = 0.0
    for i in range(len(centroids)):
        lo, hi = full_bounds[i], full_bounds[i + 1]
        c = centroids[i]
        mse_i = _integrate(lambda x: (x - c) ** 2 * pdf(x), lo, hi)
        total_mse += mse_i

    return total_mse


def get_codebook(d: int, bits: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Get precomputed codebook as torch tensors. Cached after first computation.

    Returns:
        (centroids, boundaries) as float32 tensors on the given device.
    """
    key = (d, bits)
    if key not in _CODEBOOK_CACHE:
        centroids_np, boundaries_np = compute_lloyd_max_codebook(d, bits)
        _CODEBOOK_CACHE[key] = (centroids_np, boundaries_np)

    centroids_np, boundaries_np = _CODEBOOK_CACHE[key]
    centroids = torch.tensor(centroids_np, dtype=torch.float32, device=device)
    boundaries = torch.tensor(boundaries_np, dtype=torch.float32, device=device)
    return centroids, boundaries

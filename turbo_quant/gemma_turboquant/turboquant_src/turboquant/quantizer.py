"""
TurboQuantizer: core quantize/dequantize operations.

Implements Algorithm 1 (TurboQuant_mse) from the paper:
1. Random rotation Π (QR decomposition with sign fix)
2. Scalar quantization using precomputed Lloyd-Max codebook
3. uint4 bit packing for storage
"""

import torch
from .codebook import get_codebook
from .packing import pack_uint4, unpack_uint4, pack_uint2, unpack_uint2


class TurboQuantizer:
    """Quantizes vectors on the unit hypersphere using random rotation + optimal scalar quantization.

    Each instance has its own random rotation matrix Π, enabling statistical independence
    when used per-layer.
    """

    def __init__(self, dim: int = 128, bits: int = 4, device: str = "cuda", seed: int | None = None):
        """
        Args:
            dim: Vector dimension (head_dim, typically 128).
            bits: Bits per coordinate (2 or 4).
            device: Target device.
            seed: Optional RNG seed for reproducible rotation matrix.
        """
        self.dim = dim
        self.bits = bits
        self.device = device

        # Generate random rotation matrix Π ∈ SO(d) via QR with sign convention
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        else:
            gen.seed()
        A = torch.randn(dim, dim, generator=gen)
        Q, R = torch.linalg.qr(A)
        # Sign fix: Π = Q * sign(diag(R)) ensures uniform distribution on SO(d)
        self.rotation = (Q * torch.sign(torch.diag(R))).to(torch.float32).to(device)

        # Load precomputed codebook
        centroids, boundaries = get_codebook(dim, bits, device=device)
        self.centroids = centroids  # (2^bits,) float32
        self.boundaries = boundaries  # (2^bits - 1,) float32

        # Choose pack/unpack functions based on bit-width
        if bits == 4:
            self._pack = pack_uint4
            self._unpack = unpack_uint4
        elif bits == 2:
            self._pack = pack_uint2
            self._unpack = unpack_uint2
        else:
            raise ValueError(f"Unsupported bits={bits}. Use 2 or 4.")

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize input tensor.

        Args:
            x: BF16/FP16 tensor of shape (..., dim). Vectors need NOT be unit norm —
               norms are extracted and stored separately.

        Returns:
            (packed, norms) where:
                packed: uint8 tensor of shape (..., dim // pack_factor)
                norms: BF16 tensor of shape (...,)
        """
        original_dtype = x.dtype
        # 1. Extract and store norms
        norms = x.float().norm(dim=-1)  # (...,)

        # 2. Normalize to unit sphere (avoid div by zero for zero vectors)
        x_unit = x.float() / norms.unsqueeze(-1).clamp(min=1e-8)

        # 3. Random rotation in FP32: y = x_unit @ Π^T  (equivalent to Π @ x for each vector)
        # x_unit: (..., dim), rotation: (dim, dim)
        # We want each vector rotated: y_i = Π @ x_i, which is x_unit @ Π^T
        x_rot = x_unit @ self.rotation.T  # (..., dim) FP32

        # 4. Scalar quantize: find nearest centroid for each coordinate
        indices = torch.bucketize(x_rot, self.boundaries)  # (..., dim) int64
        indices = indices.clamp(0, (2**self.bits) - 1).to(torch.uint8)

        # 5. Pack
        packed = self._pack(indices)

        return packed, norms.to(original_dtype)

    def dequantize(self, packed: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Dequantize packed indices back to approximate vectors.

        Args:
            packed: uint8 tensor from quantize().
            norms: BF16 tensor of norms from quantize().

        Returns:
            Reconstructed tensor of shape (..., dim) in the same dtype as norms.
        """
        original_dtype = norms.dtype

        # 1. Unpack indices
        indices = self._unpack(packed)  # (..., dim) uint8

        # 2. Lookup centroids
        x_rot_approx = self.centroids[indices.long()]  # (..., dim) float32

        # 3. Inverse rotation in FP32: x_approx = x_rot_approx @ Π
        x_unit_approx = x_rot_approx @ self.rotation  # (..., dim) FP32

        # 4. Rescale by stored norms
        x_approx = norms.float().unsqueeze(-1) * x_unit_approx

        return x_approx.to(original_dtype)

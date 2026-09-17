"""
Bit packing utilities for uint4 and uint2 quantized indices.

uint4: 2 values per byte (128 dims → 64 bytes)
uint2: 4 values per byte (128 dims → 32 bytes)
"""

import torch


def pack_uint4(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint8 tensor with values 0-15 into uint4 format (2 values per byte).

    Args:
        indices: uint8 tensor with shape (..., d) where d is even.
                 Values must be in [0, 15].

    Returns:
        uint8 tensor with shape (..., d // 2).
    """
    assert indices.shape[-1] % 2 == 0, f"Last dim must be even, got {indices.shape[-1]}"
    high = indices[..., 0::2] << 4
    low = indices[..., 1::2]
    return (high | low).to(torch.uint8)


def unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint4 format back to uint8 tensor with values 0-15.

    Args:
        packed: uint8 tensor with shape (..., d // 2).

    Returns:
        uint8 tensor with shape (..., d) where d = 2 * packed.shape[-1].
    """
    high = packed >> 4
    low = packed & 0x0F
    # Interleave: [h0, l0, h1, l1, ...]
    d_half = packed.shape[-1]
    out = torch.stack([high, low], dim=-1)  # (..., d_half, 2)
    return out.reshape(*packed.shape[:-1], d_half * 2)


def pack_uint2(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint8 tensor with values 0-3 into uint2 format (4 values per byte).

    Args:
        indices: uint8 tensor with shape (..., d) where d is divisible by 4.
                 Values must be in [0, 3].

    Returns:
        uint8 tensor with shape (..., d // 4).
    """
    assert indices.shape[-1] % 4 == 0, f"Last dim must be divisible by 4, got {indices.shape[-1]}"
    b0 = indices[..., 0::4] << 6
    b1 = indices[..., 1::4] << 4
    b2 = indices[..., 2::4] << 2
    b3 = indices[..., 3::4]
    return (b0 | b1 | b2 | b3).to(torch.uint8)


def unpack_uint2(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint2 format back to uint8 tensor with values 0-3.

    Args:
        packed: uint8 tensor with shape (..., d // 4).

    Returns:
        uint8 tensor with shape (..., d) where d = 4 * packed.shape[-1].
    """
    b0 = (packed >> 6) & 0x03
    b1 = (packed >> 4) & 0x03
    b2 = (packed >> 2) & 0x03
    b3 = packed & 0x03
    d_quarter = packed.shape[-1]
    out = torch.stack([b0, b1, b2, b3], dim=-1)  # (..., d_quarter, 4)
    return out.reshape(*packed.shape[:-1], d_quarter * 4)

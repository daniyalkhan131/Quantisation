"""
TurboQuant KV cache integration with HuggingFace Transformers.

TurboQuantLayer extends QuantizedLayer, implementing _quantize() and _dequantize()
with TurboQuant's random rotation + optimal scalar quantization.

TurboQuantCache is a Cache container that creates TurboQuantLayer instances.
"""

import torch
from transformers.cache_utils import QuantizedLayer, DynamicLayer, Cache
from transformers import PreTrainedConfig

from .quantizer import TurboQuantizer


class TurboQuantLayer(QuantizedLayer):
    """A single layer's quantized KV cache using TurboQuant.

    Each layer has its own TurboQuantizer (with its own rotation matrix Π),
    providing statistical independence between layers.
    """

    def __init__(
        self,
        dim: int = 128,
        nbits: int = 4,
        residual_length: int = 128,
        device: str = "cuda",
        layer_seed: int | None = None,
    ):
        super().__init__(
            nbits=nbits,
            axis_key=0,
            axis_value=0,
            q_group_size=dim,
            residual_length=residual_length,
        )
        self.quantizer = TurboQuantizer(dim=dim, bits=nbits, device=device, seed=layer_seed)

    def _quantize(self, tensor: torch.Tensor, axis: int) -> tuple[torch.Tensor, torch.Tensor]:
        packed, norms = self.quantizer.quantize(tensor)
        return (packed, norms)

    def _dequantize(self, q_tensor: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        packed, norms = q_tensor
        return self.quantizer.dequantize(packed, norms)


class TurboQuantCache(Cache):
    """KV cache using TurboQuant compression.

    Drop-in replacement for DynamicCache. Compresses KV cache ~4x at 4-bit
    with near-zero quality loss, using random rotation + optimal scalar quantization.

    Some transformer layers (especially layer 0) have anomalously large KV norms.
    The `skip_layers` parameter keeps these in full BF16 to preserve quality.
    A calibration pass can auto-detect which layers to skip.

    Usage:
        cache = TurboQuantCache(model.config, nbits=4)
        output = model.generate(input_ids, past_key_values=cache)
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        nbits: int = 4,
        residual_length: int = 128,
        device: str = "cuda",
        base_seed: int = 42,
        skip_layers: set[int] | None = None,
    ):
        """
        Args:
            config: Model config (needs num_hidden_layers and hidden_size/num_attention_heads).
            nbits: Bits per coordinate (2 or 4).
            residual_length: Number of recent tokens kept in full precision before quantizing.
            device: Target device.
            base_seed: Base seed for rotation matrices. Layer i uses seed = base_seed + i.
            skip_layers: Layer indices to keep in full precision (no quantization).
                         Set to {0} to skip layer 0 which often has outlier key norms.
                         If None, defaults to {0} as a safe default.
        """
        text_config = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
        num_layers = text_config.num_hidden_layers
        # Some models (e.g., Gemma-2) have explicit head_dim that differs from hidden_size/num_heads
        head_dim = getattr(text_config, "head_dim", None) or (text_config.hidden_size // text_config.num_attention_heads)

        if skip_layers is None:
            skip_layers = {0}  # Layer 0 typically has outlier key norms

        layers = []
        for i in range(num_layers):
            if i in skip_layers:
                layers.append(DynamicLayer())
            else:
                layers.append(
                    TurboQuantLayer(
                        dim=head_dim,
                        nbits=nbits,
                        residual_length=residual_length,
                        device=device,
                        layer_seed=base_seed + i,
                    )
                )
        super().__init__(layers=layers)

    @staticmethod
    def calibrate_skip_layers(
        model,
        tokenizer,
        calibration_text: str = "The quick brown fox jumps over the lazy dog.",
        norm_threshold: float = 5.0,
    ) -> set[int]:
        """Auto-detect which layers have outlier KV norms and should skip quantization.

        Runs a single forward pass and identifies layers where key norms exceed
        `norm_threshold` times the median key norm across all layers.

        Returns:
            Set of layer indices to skip.
        """
        inputs = tokenizer(calibration_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(inputs.input_ids, use_cache=True)

        cache = out.past_key_values
        norms = []
        for i in range(len(cache.layers)):
            k = cache.layers[i].keys
            if k is not None and k.numel() > 0:
                norms.append(k.float().norm(dim=-1).mean().item())
            else:
                norms.append(0.0)

        median_norm = sorted(norms)[len(norms) // 2]
        skip = {i for i, n in enumerate(norms) if n > norm_threshold * median_norm}
        return skip

import math
import torch
import torch.nn as nn
from typing import Iterable


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 8, dropout: float = 0.0, out_mask: torch.Tensor = None):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.weight = base.weight
        self.bias = base.bias
        self.use_bias = self.bias is not None
        self.weight.requires_grad = False
        if self.use_bias:
            self.bias.requires_grad = False
        self.register_buffer("out_mask", out_mask if out_mask is not None else None, persistent=False)
        if r > 0:
            self.lora_A = nn.Linear(self.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, self.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.lora_A is not None:
            delta = self.lora_B(self.lora_A(self.drop(x)))
            if self.out_mask is not None:
                delta = delta * self.out_mask
            out = out + self.scaling * delta
        return out

    @property
    def lora_parameters(self) -> Iterable[nn.Parameter]:
        if self.lora_A is None:
            return []
        return list(self.lora_A.parameters()) + list(self.lora_B.parameters())


def _replace_linear_with_lora(module: nn.Module, r: int, alpha: int, names: set, qk_only: bool = False):
    for name, child in list(module.named_children()):
        full_name = name
        # replace target Linear
        if isinstance(child, nn.Linear) and full_name in names:
            out_mask = None
            if qk_only and full_name == "qkv":
                # only apply to the first 2/3 output channels (q and k)
                of = child.out_features
                d = of // 3
                mask = torch.zeros(of)
                mask[: 2 * d] = 1.0
                out_mask = mask.view(1, -1)  # broadcast on last dim
            lora = LoRALinear(child, r=r, alpha=alpha, out_mask=out_mask)
            setattr(module, name, lora)
        else:
            _replace_linear_with_lora(child, r, alpha, names, qk_only=qk_only)


def apply_lora_to_aggregator(aggregator: nn.Module, r: int = 8, alpha: int = 8, target: str = "all"):
    """
    Inject LoRA into aggregator blocks:
    target:
      - "all": qkv, proj, fc1, fc2
      - "attn": qkv, proj
      - "qkv": qkv only
      - "qk": qkv with LoRA only on q,k (masking out v)
      - "mlp": fc1, fc2
    """
    if target == "all":
        names = {"qkv", "proj", "fc1", "fc2"}
        qk_only = False
    elif target == "attn":
        names = {"qkv", "proj"}
        qk_only = False
    elif target == "qkv":
        names = {"qkv"}
        qk_only = False
    elif target == "qk":
        names = {"qkv"}
        qk_only = True
    elif target == "mlp":
        names = {"fc1", "fc2"}
        qk_only = False
    else:
        names = {"qkv", "proj", "fc1", "fc2"}
        qk_only = False
    for blk in getattr(aggregator, "frame_blocks", []):
        _replace_linear_with_lora(blk, r, alpha, names, qk_only=qk_only)
    for blk in getattr(aggregator, "global_blocks", []):
        _replace_linear_with_lora(blk, r, alpha, names, qk_only=qk_only)


def mark_only_lora_trainable(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False
    for m in module.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora_parameters:
                p.requires_grad = True

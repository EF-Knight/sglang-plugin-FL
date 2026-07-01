# GCU FusedMoE operator implementation.

from __future__ import annotations

import torch


def fused_moe_gcu(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    from sglang.srt.layers.moe.fused_moe_native import (
        fused_moe_forward_native,
    )
    return fused_moe_forward_native(
        layer=layer,
        dispatch_output=dispatch_output,
    )
    
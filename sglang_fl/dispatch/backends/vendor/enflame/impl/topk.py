# GCU TopK operator implementation.

from __future__ import annotations

from typing import Optional

import torch


def topk_gcu(
    obj,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info=None,
):
    
    from sglang.srt.layers.moe.topk import select_experts

    obj.topk_config.torch_native = True
    return select_experts(
        hidden_states=hidden_states,
        layer_id=obj.layer_id,
        router_logits=router_logits,
        topk_config=obj.topk_config,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )

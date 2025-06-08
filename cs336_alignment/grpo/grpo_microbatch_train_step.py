from cs336_alignment.grpo_utils import compute_policy_gradient_loss, masked_mean
from cs336_alignment.utils import masked_normalize
import torch
from typing import Literal


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal[
        "no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"
    ],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    normalize_mode: Literal["mean", "constant", "microbatch"] = "mean",
    normalize_constant: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    policy_gradient_loss, meta = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )

    if normalize_mode == "mean":
        loss = (
            masked_mean(policy_gradient_loss, response_mask, dim=-1).mean()
            / gradient_accumulation_steps
        )
    elif normalize_mode in ["constant", "microbatch"]:
        assert normalize_constant is not None

        if normalize_mode == "constant":
            constant = normalize_constant
        elif normalize_mode == "microbatch":
            # Normalize by longest sequence in microbatch
            constant = response_mask.sum(dim=-1).max().item()

        loss = (
            masked_normalize(
                policy_gradient_loss, response_mask, constant, dim=-1
            ).mean()
            / gradient_accumulation_steps
        )

    loss.backward()
    detached_loss = loss.detach()

    # Trying to avoid OOM; haven't ablated this to confirm that it helps
    del loss
    del policy_gradient_loss

    return detached_loss, meta

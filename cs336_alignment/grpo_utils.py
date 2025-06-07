from collections.abc import Callable

import torch
from einops import rearrange
from typing import Literal


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    # List of dicts of rewards (format_reward, answer_reward, reward) for each rollout
    rewards = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]

    # Tensor of aggregate reward for each rollout
    raw_rewards = torch.tensor([reward["reward"] for reward in rewards])

    # Reshape into groups for normalization
    reward_groups = rearrange(
        raw_rewards,
        "(n_groups group_size) -> n_groups group_size",
        group_size=group_size,
    )

    # Compute group statistics
    group_means = reward_groups.mean(dim=-1)  # shape: (n_groups,)
    group_stds = reward_groups.std(dim=-1)  # shape: (n_groups,)

    # Prepare normalization divisors
    divisors = (
        group_stds + advantage_eps if normalize_by_std else torch.ones_like(group_stds)
    )

    # Compute advantages on grouped data, then flatten
    group_advantages = (reward_groups - group_means.unsqueeze(-1)) / divisors.unsqueeze(
        -1
    )

    advantages = rearrange(
        group_advantages, "n_groups group_size -> (n_groups group_size)"
    )

    return advantages, raw_rewards, {"rewards_list": rewards}


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, policy_log_probs: torch.Tensor
) -> torch.Tensor:
    return -policy_log_probs * raw_rewards_or_advantages


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    clip: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Compute ratio in log space and exponentiate out
    ratios = torch.exp(policy_log_probs - old_log_probs)
    scores = ratios * advantages

    mean_ratio = ratios.mean()

    if not clip:
        return -scores, {"clip_fraction": 0.0, "mean_ratio": mean_ratio}

    clipped_ratios = torch.clip(ratios, 1 - cliprange, 1 + cliprange)
    clipped_scores = clipped_ratios * advantages
    clip_fraction = (~torch.isclose(scores, clipped_scores)).float().mean()

    return -torch.minimum(scores, clipped_scores), {
        "clip_fraction": clip_fraction,
        "mean_ratio": mean_ratio,
    }


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal[
        "no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"
    ],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert loss_type in (
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
        "grpo_no_clip",
    )

    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}

    assert advantages is not None

    if loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}

    if loss_type == "grpo_clip":
        assert old_log_probs is not None
        assert cliprange is not None

        return compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )

    if loss_type == "grpo_no_clip":
        assert old_log_probs is not None

        return compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange, clip=False
        )


def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    # return torch.masked.mean(tensor, dim, mask=mask)
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim) / mask.sum(dim)

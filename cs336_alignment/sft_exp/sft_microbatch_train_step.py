import torch
from cs336_alignment.utils import masked_normalize


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Mask and normalize, summing along sequence dimension
    masked_normalized_probs = masked_normalize(
        policy_log_probs, response_mask, normalize_constant, dim=-1
    )

    # Mean loss across sequences
    loss = -masked_normalized_probs.mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()

    return loss, {}

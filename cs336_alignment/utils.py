from tokenizers import Tokenizer
from transformers import PreTrainedModel
from einops import rearrange
from vllm import LLM, SamplingParams
from collections.abc import Callable
from cs336_alignment.vllm import EvalResult
import torch


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: Tokenizer
) -> dict[str, torch.Tensor]:
    assert len(prompt_strs) == len(output_strs)

    prompt_tokens_list = []
    output_tokens_list = []
    concated_tokens_list = []

    for prompt_str, output_str in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer.encode(
            prompt_str,
        )
        output_tokens = tokenizer.encode(output_str)

        prompt_tokens_list.append(prompt_tokens)
        output_tokens_list.append(output_tokens)
        concated_tokens_list.append(prompt_tokens + output_tokens)

    max_len = max(len(t) for t in concated_tokens_list)

    input_ids = torch.full(
        (len(concated_tokens_list), max_len - 1), tokenizer.pad_token_id
    )
    labels = torch.full(
        (len(concated_tokens_list), max_len - 1), tokenizer.pad_token_id
    )
    response_mask = torch.zeros(
        (len(concated_tokens_list), max_len - 1), dtype=torch.bool
    )

    for i, tokens in enumerate(concated_tokens_list):
        n_tokens = min(len(tokens), max_len - 1)

        prompt_tokens = prompt_tokens_list[i]
        input_ids[i, :n_tokens] = torch.tensor(tokens[:n_tokens])
        labels[i, : len(tokens) - 1] = torch.tensor(tokens[1:])
        response_mask[i, len(prompt_tokens) - 1 : len(tokens) - 1] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -torch.sum(probs * log_probs, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits

    all_log_probs = torch.log_softmax(logits, dim=-1)  # b t v
    labels_idx = rearrange(labels, "b t -> b t 1")

    log_probs_of_labels = torch.gather(all_log_probs, dim=-1, index=labels_idx)  # b t 1
    log_probs_of_labels = log_probs_of_labels.squeeze(dim=-1)  # b t

    result = {"log_probs": log_probs_of_labels}

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    masked = tensor * mask
    return masked.sum(dim) / normalize_constant


def log_generations(
    vllm_model: LLM,
    prompts: list[str],
    ground_truths: list[str],
    log_fn: Callable[[list[EvalResult]], None],
    reward_fn: Callable[[str, str], dict[str, float]] | None = None,
    sampling_params: SamplingParams | None = None,
    out_dir: str | None = "out",
    out_file: str | None = None,
) -> list[EvalResult]:
    """
    Generate from the model and log metrics using log_fn
    """
    # TODO: Implement this (optionally change signature)
    pass

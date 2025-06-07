import os
import torch
from unittest.mock import patch
from transformers import PreTrainedModel
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from cs336_alignment.common import ordered_filename
from collections.abc import Callable
from vllm import LLM, SamplingParams
from pydantic import BaseModel


class EvalMetrics(BaseModel):
    n_examples: int
    n_format_correct: int
    n_format_incorrect: int
    n_answer_correct: int
    n_answer_incorrect: int
    n_correct: int
    n_incorrect: int
    format_accuracy: float
    answer_accuracy: float
    accuracy: float


class EvalResult(BaseModel):
    prompt: str
    completion: str
    ground_truth: str
    rewards: dict[str, float]


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams | None = None,
    out_dir: str | None = "out",
    out_file: str | None = None,
    write: bool = True,
    min_tokens: int = 0,
) -> tuple[list[EvalResult], EvalMetrics]:
    """
    Eval LM on prompts, compute eval metrics, optionally serialize to disk, return evaluation results.
    """
    sampling_params = eval_sampling_params or SamplingParams(
        temperature=1.0,
        top_p=1.0,
        min_tokens=min_tokens,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    outputs = vllm_model.generate(
        prompts,
        sampling_params,
    )

    os.makedirs(out_dir, exist_ok=True)
    filename = out_file or f"{ordered_filename('eval')}.jsonl"
    outpath = os.path.join(out_dir, filename)

    results = []

    metrics = EvalMetrics(
        n_examples=len(prompts),
        n_format_correct=0,
        n_format_incorrect=0,
        n_answer_correct=0,
        n_answer_incorrect=0,
        n_correct=0,
        n_incorrect=0,
        format_accuracy=0.0,
        answer_accuracy=0.0,
        accuracy=0.0,
    )

    for i, output in enumerate(outputs):
        prompt = output.prompt
        completion = output.outputs[0].text
        ground_truth = ground_truths[i]
        rewards = reward_fn(completion, ground_truth)

        format_correct = rewards["format_reward"]
        answer_correct = rewards["answer_reward"]
        correct = rewards["reward"]

        metrics.n_format_correct += int(format_correct)
        metrics.n_format_incorrect += int(not format_correct)
        metrics.n_answer_correct += int(answer_correct)
        metrics.n_answer_incorrect += int(not answer_correct)
        metrics.n_correct += int(correct)
        metrics.n_incorrect += int(not correct)

        result = EvalResult(
            prompt=prompt,
            completion=completion,
            ground_truth=ground_truth,
            rewards=rewards,
        )

        results.append(result)

    metrics.format_accuracy = metrics.n_format_correct / metrics.n_examples
    metrics.answer_accuracy = metrics.n_answer_correct / metrics.n_examples
    metrics.accuracy = metrics.n_correct / metrics.n_examples

    if write:
        with open(outpath, "w") as f:
            f.write(metrics.model_dump_json() + "\n")
            f.write("\n".join([result.model_dump_json() for result in results]) + "\n")

    return results, metrics


def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
):
    """
    Start the inference process; here we use vLLM to hold a model on a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()

    # Handle compiled models by stripping _orig_mod prefix
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {
            key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()
        }

    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

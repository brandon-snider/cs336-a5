import os
import json
from cs336_alignment.common import ordered_filename
from collections.abc import Callable
from vllm import LLM, SamplingParams


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams | None = None,
    out_dir: str | None = "out",
    out_file: str | None = None,
) -> str:
    """
    Eval LM on prompts, compute eval metrics, serialize to disk, return path to results.
    """
    sampling_params = eval_sampling_params or SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"]
    )

    outputs = vllm_model.generate(prompts, sampling_params)

    os.makedirs(out_dir, exist_ok=True)
    filename = out_file or f"{ordered_filename('eval')}.jsonl"
    outpath = os.path.join(out_dir, filename)
    with open(outpath, "w") as fout:
        for i, output in enumerate(outputs):
            prompt = output.prompt
            completion = output.outputs[0].text
            ground_truth = ground_truths[i]
            rewards = reward_fn(completion, ground_truth)

            result = {
                "prompt": prompt,
                "completion": completion,
                "ground_truth": ground_truths[i],
                "rewards": rewards,
            }

            fout.write(json.dumps(result) + "\n")

    return outpath

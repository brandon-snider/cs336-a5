import os
import argparse
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM
from cs336_alignment.vllm import evaluate_vllm

PROMPT_TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt"

DEFAULT_EXAMPLES_PATH = "/data/a5-alignment/MATH/validation.jsonl"
DEFAULT_MODEL_PATH = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"

DEFAULT_OUT_DIR = "out"
DEFAULT_OUT_FILE = "math_baseline.jsonl"


def main(
    examples_path: str, model_path: str, out_dir: str | os.PathLike, out_file: str
):
    with open(examples_path) as f:
        examples = [json.loads(line) for line in f]

    with open(PROMPT_TEMPLATE_PATH) as f:
        prompt_template = f.read()

    prompts = [prompt_template.replace("{question}", ex["problem"]) for ex in examples]
    solutions = [ex["solution"] for ex in examples]

    evaluate_vllm(
        LLM(model=model_path),
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=solutions,
        out_dir=out_dir,
        out_file=out_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-path", default=DEFAULT_EXAMPLES_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-file", default=DEFAULT_OUT_FILE)
    args = parser.parse_args()
    main(
        examples_path=args.examples_path,
        model_path=args.model_path,
        out_dir=args.out_dir,
        out_file=args.out_file,
    )

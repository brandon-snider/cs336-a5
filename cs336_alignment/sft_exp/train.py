import argparse
import logging
import json
from pathlib import Path
import random
import submitit
from omegaconf import OmegaConf

from cs336_alignment.vllm import (
    evaluate_vllm,
    init_vllm,
    load_policy_into_vllm_instance,
)

import torch
from tqdm import trange
import wandb
from cs336_alignment.sft_exp.sft_microbatch_train_step import sft_microbatch_train_step
from cs336_alignment.utils import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.sft_exp.config.defaults import Config
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

logger = logging.getLogger(__name__)


def get_batch(
    tokenized_examples: dict[str, torch.Tensor], batch_size: int, device: str
) -> dict[str, torch.Tensor]:
    """Get a batch of examples from the tokenized examples."""
    batch_indices = random.sample(
        range(len(tokenized_examples["input_ids"])), batch_size
    )
    return {k: v[batch_indices].to(device) for k, v in tokenized_examples.items()}


def main(cfg: Config):
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(cfg.training.seed)
    random.seed(cfg.training.seed)

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[cfg.training.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        cfg.paths.model_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        device_map=cfg.training.device,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.model_path)

    logger.info("Tokenizing train examples...")
    with open(cfg.paths.train_examples_path, "r") as f:
        train_examples = [json.loads(line) for line in f]
        if cfg.training.max_unique_examples is not None:
            train_examples = train_examples[: cfg.training.max_unique_examples]

        if cfg.training.correct_only:
            train_examples = [
                example
                for example in train_examples
                if r1_zero_reward_fn(example["response"], example["ground_truth"])[
                    "reward"
                ]
            ]

        logger.info(
            f"Using {len(train_examples)} train examples {'(correct only)' if cfg.training.correct_only else ''}"
        )

        train_tokenized = tokenize_prompt_and_output(
            [example["prompt"] for example in train_examples],
            [example["response"] for example in train_examples],
            tokenizer,
        )

    with (
        open(cfg.paths.val_examples_path, "r") as f_val,
        open(cfg.paths.val_prompt_template_path, "r") as f_prompt,
    ):
        prompt_template = f_prompt.read()
        val_examples = [json.loads(line) for line in f_val]
        val_prompts = [
            prompt_template.replace("{question}", ex["problem"]) for ex in val_examples
        ]
        val_answers = [ex["answer"] for ex in val_examples]

    amp_ctx = torch.amp.autocast(
        device_type=cfg.training.device,
        dtype=torch_dtype,
    )

    if cfg.training.compile:
        model = torch.compile(model)

    vllm_model = init_vllm(
        cfg.paths.model_path.as_posix(), cfg.training.vllm_device, cfg.training.seed
    )

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
    params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": params_to_decay, "weight_decay": cfg.training.weight_decay},
        {"params": params_to_not_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        eps=cfg.training.adam_eps,
        fused=True,
    )

    if cfg.training.override_steps_with_epochs:
        cfg.training.train_steps = (
            len(train_examples)
            * cfg.training.train_epochs
            // cfg.training.train_batch_size
        )

    warmup_steps = int(cfg.training.warmup_ratio * cfg.training.train_steps)

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=cfg.training.train_steps,
    )

    use_wandb = cfg.training.wandb_project and cfg.training.wandb_entity

    if use_wandb:
        wandb.init(
            entity=cfg.training.wandb_entity,
            project=cfg.training.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.paths.model_output.name,
        )

        # Setup wandb metrics
        wandb.define_metric("train_step")  # the x‑axis for training
        wandb.define_metric("eval_step")  # the x‑axis for evaluation
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    cfg.paths.model_output.mkdir(parents=True, exist_ok=True)

    batch = get_batch(
        train_tokenized, cfg.training.train_microbatch_size, cfg.training.device
    )

    model.train()

    def evaluate(eval_step: int, n_examples: int | None = None):
        logger.info(f"Evaluating at step {eval_step:,}...")

        val_prompts_subset = val_prompts
        val_answers_subset = val_answers

        if n_examples is not None:
            indices = random.sample(range(len(val_prompts)), n_examples)
            val_prompts_subset = [val_prompts[i] for i in indices]
            val_answers_subset = [val_answers[i] for i in indices]

        load_policy_into_vllm_instance(model, vllm_model)
        eval_results, eval_metrics = evaluate_vllm(
            vllm_model,
            reward_fn=r1_zero_reward_fn,
            prompts=val_prompts_subset,
            ground_truths=val_answers_subset,
            write=False,
        )

        if use_wandb:
            wandb.log(
                {
                    "eval/format_accuracy": eval_metrics.format_accuracy,
                    "eval/answer_accuracy": eval_metrics.answer_accuracy,
                    "eval/accuracy": eval_metrics.accuracy,
                    "eval_step": eval_step,
                }
            )

        logger.info(
            f"Accuracy (step {eval_step:,}): {eval_metrics.accuracy:.3f} (format: {eval_metrics.format_accuracy:.3f}, answer: {eval_metrics.answer_accuracy:.3f})"
        )

    if cfg.training.eval_before_training:
        logger.info("Evaluating initial model...")
        evaluate(0, cfg.training.eval_n_examples)

    eval_step = 0
    eval_interval = (
        cfg.training.eval_interval
        or cfg.training.train_steps // cfg.training.eval_steps
    )

    for step in (pbar := trange(cfg.training.train_steps, desc="Training")):
        accumulated_entropy = 0.0

        for _ in range(cfg.training.gradient_accumulation_steps):
            with amp_ctx:
                log_probs_result = get_response_log_probs(
                    model,
                    batch["input_ids"],
                    batch["labels"],
                    return_token_entropy=True,
                )

                log_probs = log_probs_result["log_probs"]
                token_entropy = log_probs_result["token_entropy"]
                accumulated_entropy += token_entropy.mean().item()

                next_batch = get_batch(
                    train_tokenized,
                    cfg.training.train_microbatch_size,
                    cfg.training.device,
                )

                loss, _ = sft_microbatch_train_step(
                    log_probs,
                    batch["response_mask"],
                    cfg.training.gradient_accumulation_steps,
                )

            batch = next_batch

        avg_token_entropy = (
            accumulated_entropy / cfg.training.gradient_accumulation_steps
        )
        logged_loss = loss.item() * cfg.training.gradient_accumulation_steps

        if cfg.training.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.max_grad_norm
            )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()

        pbar.set_postfix(
            {
                "lr": lr_scheduler.get_last_lr()[0],
                "loss": logged_loss,
            }
        )

        if use_wandb and step % cfg.training.log_interval == 0:
            wandb.log(
                {
                    "train/loss": logged_loss,
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/avg_token_entropy": avg_token_entropy,
                    "train_step": step,
                }
            )

        if step > 0 and step % eval_interval == 0:
            evaluate(eval_step, cfg.training.eval_n_examples)
            eval_step += 1

    # Evaluate on the entire validation set at the end
    evaluate(eval_step)

    logger.info(f"Saving model weights to {cfg.paths.model_output}")
    model.save_pretrained(cfg.paths.model_output)
    tokenizer.save_pretrained(cfg.paths.model_output)

    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    default_config_dir = "cs336_alignment/sft_exp/config"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, default=default_config_dir)
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    default_cfg = OmegaConf.structured(Config)

    if args.config_name is not None:
        override_cfg = OmegaConf.load(Path(args.config_dir) / f"{args.config_name}")
        cfg = OmegaConf.merge(default_cfg, override_cfg)
    else:
        cfg = default_cfg

    executor = submitit.AutoExecutor(
        folder="/data/c-sniderb/a5-alignment/sft-experiment/slurm"
    )
    executor.update_parameters(
        timeout_min=60,
        slurm_account="student",
        slurm_partition="a4-batch",
        slurm_qos="a4-batch-qos",
        slurm_gpus_per_node="2",
    )

    if args.submit:
        job = executor.submit(main, cfg)
        logger.info(f"Submitted job with ID {job.job_id}")
        if args.wait:
            job.result()
    else:
        main(cfg)

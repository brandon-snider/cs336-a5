import argparse
import logging
import json
from pathlib import Path
import random
import submitit
from omegaconf import OmegaConf
from vllm import SamplingParams

from cs336_alignment.grpo_utils import compute_group_normalized_rewards
from cs336_alignment.vllm import (
    EvalResult,
    evaluate_vllm,
    init_vllm,
    load_policy_into_vllm_instance,
)

import torch
from tqdm import trange
import wandb
from cs336_alignment.utils import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.grpo.config.defaults import Config
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from transformers import AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.common import default_sampling_params
from cs336_alignment.grpo.grpo_microbatch_train_step import grpo_microbatch_train_step

NORMALIZE_CONSTANT = 1024

logger = logging.getLogger(__name__)


def main(cfg: Config):
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    assert (
        cfg.training.train_batch_size % cfg.training.gradient_accumulation_steps == 0
    ), "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = (
        cfg.training.train_batch_size // cfg.training.gradient_accumulation_steps
    )

    assert cfg.training.rollout_batch_size % cfg.training.group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = (
        cfg.training.rollout_batch_size // cfg.training.group_size
    )

    assert cfg.training.train_batch_size >= cfg.training.group_size, (
        "train_batch_size must be greater than or equal to group_size"
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

    with open(cfg.paths.prompt_template_path, "r") as f:
        prompt_template = f.read()

    with open(cfg.paths.train_examples_path, "r") as f:
        grpo_examples = [json.loads(line) for line in f]
        logger.info(f"Using {len(grpo_examples)} GRPO examples")

    with open(cfg.paths.val_examples_path, "r") as f:
        val_examples = [json.loads(line) for line in f]
        val_prompts = [
            prompt_template.replace("{question}", ex["problem"]) for ex in val_examples
        ]
        val_answers = [ex["answer"] for ex in val_examples]

    amp_ctx = torch.amp.autocast(
        device_type=cfg.training.device,
        dtype=torch_dtype,
    )

    if cfg.training.torch_compile:
        model = torch.compile(model)

    vllm_model = init_vllm(
        cfg.paths.model_path.as_posix(),
        cfg.training.vllm_device,
        cfg.training.seed,
        gpu_memory_utilization=cfg.training.gpu_memory_utilization,
    )

    use_wandb = cfg.training.wandb_project and cfg.training.wandb_entity

    if use_wandb:
        wandb.init(
            entity=cfg.training.wandb_entity,
            project=cfg.training.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.paths.model_output.name,
            tags=cfg.training.wandb_tags,
        )

        # Setup wandb metrics
        wandb.define_metric("train_step")  # the x‑axis for training
        wandb.define_metric("eval_step")  # the x‑axis for evaluation
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    cfg.paths.model_output.mkdir(parents=True, exist_ok=True)

    reward_fn = (
        question_only_reward_fn if cfg.training.question_only else r1_zero_reward_fn
    )

    def evaluate(
        eval_step: int, n_examples: int | None = None, grpo_step: int | None = None
    ):
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
            reward_fn=reward_fn,
            prompts=val_prompts_subset,
            ground_truths=val_answers_subset,
            write=False,
            min_tokens=cfg.training.sampling_min_tokens,
        )

        if use_wandb:
            wandb.log(
                {
                    "eval/format_accuracy": eval_metrics.format_accuracy,
                    "eval/answer_accuracy": eval_metrics.answer_accuracy,
                    "eval/accuracy": eval_metrics.accuracy,
                    "eval_step": eval_step,
                    "grpo_step": grpo_step,
                }
            )

        logger.info(
            f"Accuracy (step {eval_step:,}): {eval_metrics.accuracy:.3f} (format: {eval_metrics.format_accuracy:.3f}, answer: {eval_metrics.answer_accuracy:.3f})"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        eps=cfg.training.adam_eps,
        fused=True,
    )

    eval_step = 0
    grad_descent_step = 0

    if cfg.training.eval_before_training:
        logger.info("Evaluating initial model...")
        evaluate(eval_step, cfg.training.eval_n_examples)
        eval_step += 1

    # OUTER LOOP: GRPO

    for grpo_step in range(cfg.training.n_grpo_steps):
        grpo_batch = random.sample(grpo_examples, n_prompts_per_rollout_batch)

        repeated_rollout_prompts = [
            prompt_template.replace("{question}", ex["problem"])
            for _ in range(cfg.training.group_size)
            for ex in grpo_batch
        ]
        repeated_ground_truths = [
            ex["answer"] for _ in range(cfg.training.group_size) for ex in grpo_batch
        ]

        logger.info(
            f"Generating {cfg.training.group_size} * {n_prompts_per_rollout_batch:,} = {len(repeated_rollout_prompts):,} rollouts for GRPO step {grpo_step}..."
        )

        sampling_params = SamplingParams(
            temperature=cfg.training.sampling_temperature,
            top_p=cfg.training.sampling_top_p,
            max_tokens=cfg.training.sampling_max_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            min_tokens=cfg.training.sampling_min_tokens,
        )

        # Generate rollouts
        load_policy_into_vllm_instance(model, vllm_model)
        grpo_batch_outputs = vllm_model.generate(
            repeated_rollout_prompts,
            sampling_params,
        )

        rollout_responses = [output.outputs[0].text for output in grpo_batch_outputs]

        # Compute advantages
        rollout_advantages, rollout_raw_rewards, rollout_rewards_meta = (
            compute_group_normalized_rewards(
                reward_fn,
                rollout_responses,
                repeated_ground_truths,
                cfg.training.group_size,
                cfg.training.advantage_eps,
                normalize_by_std=cfg.training.use_std_normalization,
            )
        )

        rollout_advantages = rollout_advantages.unsqueeze(-1).to(cfg.training.device)
        rollout_raw_rewards = rollout_raw_rewards.unsqueeze(-1).to(cfg.training.device)
        rollout_rewards_list = rollout_rewards_meta["rewards_list"]

        # Prepare training examples
        train_prompts = []
        train_outputs = []

        rollouts_to_log = []

        for i, rollout_output in enumerate(grpo_batch_outputs):
            rollout_to_log = EvalResult(
                prompt=rollout_output.prompt,
                completion=rollout_output.outputs[0].text,
                ground_truth=repeated_ground_truths[i],
                rewards={},
            )
            rollouts_to_log.append(rollout_to_log)

            train_prompts.append(rollout_output.prompt)
            train_outputs.append(rollout_output.outputs[0].text)

        with open(
            cfg.paths.model_output / f"rollouts_grpo_step_{grpo_step:03d}.jsonl", "w"
        ) as f:
            f.write(
                "\n".join([rollout.model_dump_json() for rollout in rollouts_to_log])
                + "\n"
            )

        logger.info(f"Tokenizing {len(train_prompts)} training examples")

        # Tokenize training examples
        train_tokenized = tokenize_prompt_and_output(
            train_prompts,
            train_outputs,
            tokenizer,
            device=cfg.training.device,
        )

        old_log_probs = None

        if cfg.training.loss_type in ("grpo_clip", "grpo_no_clip"):
            # Get old log probs
            logger.info("Getting old log probs...")

            old_log_probs_tensors = []

            for batch_idx in range(
                0,
                len(train_tokenized["input_ids"]),
                cfg.training.old_log_probs_batch_size,
            ):
                with amp_ctx, torch.inference_mode():
                    old_log_probs_batch = (
                        get_response_log_probs(
                            model,
                            train_tokenized["input_ids"][
                                batch_idx : batch_idx
                                + cfg.training.old_log_probs_batch_size
                            ],
                            train_tokenized["labels"][
                                batch_idx : batch_idx
                                + cfg.training.old_log_probs_batch_size
                            ],
                            return_token_entropy=False,
                        )["log_probs"]
                        .detach()
                        .to(cfg.training.device)
                    )

                old_log_probs_tensors.append(old_log_probs_batch)

            old_log_probs = torch.cat(old_log_probs_tensors, dim=0).detach()
            del old_log_probs_tensors

        model.train()

        logger.info(f"Training GRPO step {grpo_step}...")

        # INNER LOOP: GRPO EPOCHS OVER ROLLOUT BATCH
        for epoch in range(cfg.training.epochs_per_rollout_batch):
            # Step through train batches until we've seen all examples in the rollout batch
            for step in (
                pbar := trange(
                    cfg.training.rollout_batch_size // cfg.training.train_batch_size,
                    desc=f"GRPO step {grpo_step}: Training GRPO epoch {epoch}",
                )
            ):
                # Accumulate entropy over microbatches that comprise a single gradient descent step
                accumulated_entropy = 0.0
                accumulated_clip_fraction = 0.0
                accumulated_mean_ratio = 0.0

                batch_start_idx = step * cfg.training.train_batch_size
                batch_end_idx = batch_start_idx + cfg.training.train_batch_size

                batch_response_masks = train_tokenized["response_mask"][
                    batch_start_idx:batch_end_idx
                ]
                batch_mean_response_length = batch_response_masks.sum(dim=-1).mean(
                    dtype=torch.float32
                )

                # Step through microbatches that comprise a single gradient descent step
                for microbatch_idx in range(cfg.training.gradient_accumulation_steps):
                    base_idx = step * cfg.training.train_batch_size
                    start_idx = base_idx + microbatch_idx * micro_train_batch_size
                    end_idx = start_idx + micro_train_batch_size

                    # Get the next microbatch of input_ids, labels, and response_mask
                    microbatch = {
                        k: v[start_idx:end_idx].to(cfg.training.device)
                        for k, v in train_tokenized.items()
                    }

                    # Get current policy log_probs and token_entropy for the microbatch
                    with amp_ctx:
                        log_probs_result = get_response_log_probs(
                            model,
                            microbatch["input_ids"],
                            microbatch["labels"],
                            return_token_entropy=True,
                        )

                    current_log_probs = log_probs_result["log_probs"]
                    token_entropy = log_probs_result["token_entropy"]
                    accumulated_entropy += token_entropy.mean().item()

                    old_log_probs_microbatch = (
                        old_log_probs[start_idx:end_idx]
                        if old_log_probs is not None
                        else None
                    )

                    loss, meta = grpo_microbatch_train_step(
                        current_log_probs,
                        microbatch["response_mask"],
                        cfg.training.gradient_accumulation_steps,
                        loss_type=cfg.training.loss_type,
                        raw_rewards=rollout_raw_rewards[start_idx:end_idx],
                        advantages=rollout_advantages[start_idx:end_idx],
                        old_log_probs=old_log_probs_microbatch,
                        cliprange=cfg.training.cliprange,
                        normalize_mode=cfg.training.normalize_mode,
                        normalize_constant=NORMALIZE_CONSTANT,
                    )

                    accumulated_clip_fraction += meta.get("clip_fraction", 0.0)
                    accumulated_mean_ratio += meta.get("mean_ratio", 0.0)

                avg_token_entropy = (
                    accumulated_entropy / cfg.training.gradient_accumulation_steps
                )
                avg_clip_fraction = (
                    accumulated_clip_fraction / cfg.training.gradient_accumulation_steps
                )
                avg_mean_ratio = (
                    accumulated_mean_ratio / cfg.training.gradient_accumulation_steps
                )
                logged_loss = loss.item() * cfg.training.gradient_accumulation_steps

                max_grad_norm = (
                    cfg.training.max_grad_norm
                    if cfg.training.max_grad_norm is not None
                    else float("inf")
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                pbar.set_postfix({"loss": logged_loss})

                if (
                    use_wandb
                    and grad_descent_step % cfg.training.log_step_interval == 0
                ):
                    # Compute mean rewards for all rollouts in the current train batch
                    step_rewards_list = rollout_rewards_list[
                        step * cfg.training.train_batch_size : (step + 1)
                        * cfg.training.train_batch_size
                    ]

                    step_mean_format_reward = sum(
                        reward["format_reward"] for reward in step_rewards_list
                    ) / len(step_rewards_list)
                    step_mean_answer_reward = sum(
                        reward["answer_reward"] for reward in step_rewards_list
                    ) / len(step_rewards_list)
                    step_mean_reward = sum(
                        reward["reward"] for reward in step_rewards_list
                    ) / len(step_rewards_list)

                    wandb.log(
                        {
                            "train/loss": logged_loss,
                            "train/avg_token_entropy": avg_token_entropy,
                            "train/avg_clip_fraction": avg_clip_fraction,
                            "train/avg_mean_ratio": avg_mean_ratio,
                            "train/grad_norm": grad_norm,
                            "train/format_reward": step_mean_format_reward,
                            "train/answer_reward": step_mean_answer_reward,
                            "train/mean_response_length": batch_mean_response_length,
                            "train/reward": step_mean_reward,
                            "train_step": grad_descent_step,
                            "grpo_step": grpo_step,
                        }
                    )

                if grad_descent_step % cfg.training.eval_step_interval == 0 and (
                    grad_descent_step > 0 or not cfg.training.eval_before_training
                ):
                    logger.info(
                        f"Evaluating at grad_descent_step {grad_descent_step:,}..."
                    )
                    evaluate(eval_step, cfg.training.eval_n_examples, grpo_step)
                    eval_step += 1

                grad_descent_step += 1

        # Figure out which (if any) we actually need to delete manually
        del rollout_advantages, rollout_raw_rewards, old_log_probs, train_tokenized
        del rollout_responses, rollout_rewards_meta, rollout_rewards_list
        del grpo_batch_outputs
        torch.cuda.empty_cache()

    # Evaluate on the entire validation set at the end
    evaluate(eval_step, grpo_step=cfg.training.n_grpo_steps)

    logger.info(f"Saving model weights to {cfg.paths.model_output}")
    model.save_pretrained(cfg.paths.model_output)
    tokenizer.save_pretrained(cfg.paths.model_output)

    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    default_config_dir = "cs336_alignment/grpo/config"

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

    executor = submitit.AutoExecutor(folder="/data/c-sniderb/a5-alignment/grpo/slurm")
    executor.update_parameters(
        timeout_min=240,
        slurm_account="student",
        slurm_partition="a5-batch",
        slurm_qos="a5-batch-qos",
        slurm_gpus_per_node="1",
        slurm_nodes=1,
    )

    if args.submit:
        job = executor.submit(main, cfg)
        logger.info(f"Submitted job with ID {job.job_id}")
        if args.wait:
            job.result()
    else:
        main(cfg)

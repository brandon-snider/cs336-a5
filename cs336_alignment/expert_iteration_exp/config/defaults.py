from dataclasses import dataclass, field
from pathlib import Path
from omegaconf import OmegaConf

# Register a custom resolver for integer division
OmegaConf.register_new_resolver("div", lambda x, y: int(x) // int(y))


@dataclass
class PathsConfig:
    train_examples_path: Path = Path("/data/a5-alignment/MATH/train.jsonl")
    val_examples_path: Path = Path("/data/a5-alignment/MATH/validation.jsonl")
    prompt_template_path: Path = Path("cs336_alignment/prompts/r1_zero.prompt")
    model_path: Path = Path("/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    model_output: Path = Path("/data/c-sniderb/a5-alignment/expert-iteration-exp")


@dataclass
class TrainingConfig:
    seed: int = 42
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    vllm_device: str = "cuda:1"

    ei_steps: int = 5
    ei_batch_size: int = 256
    ei_rollouts_per_question: int = 10

    sft_epochs_per_batch: int = 10
    sft_batch_size: int = 64
    sft_grad_accum_steps: int = 8
    sft_lr: float = 5e-5

    sft_microbatch_size: int = (
        "${div:${training.sft_batch_size},${training.sft_grad_accum_steps}}"
    )

    eval_interval: int | None = None
    eval_steps_per_sft_epoch: int | None = 1
    eval_before_training: bool = True
    eval_n_examples: int | None = 1000

    max_grad_norm: float = 1.0

    warmup_steps: int = 0
    # warmup_ratio: float = 0.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    wandb_entity: str | None = "brandon-snider-stanford-university"
    wandb_project: str = "cs336-a5"
    # wandb_project: str | None = None
    compile: bool = True
    log_interval: int = 1


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

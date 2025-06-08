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
    model_output: Path = Path("/data/c-sniderb/a5-alignment/grpo-experiments")


@dataclass
class TrainingConfig:
    seed: int = 42
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    vllm_device: str = "cuda:0"

    n_grpo_steps: int = 200
    learning_rate: float = 2e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    epochs_per_rollout_batch: int = 1  # On-policy
    train_batch_size: int = 256  # On-policy
    gradient_accumulation_steps: int = 128  # microbatch size is 2, will fit on H100

    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4  # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    sampling_top_p: float = 1.0

    gpu_memory_utilization: float = 0.2

    # Have to retry
    loss_type: str = "reinforce_with_baseline"  # "no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"

    use_std_normalization: bool = True

    # - "mean" (div. each by seq's adv. by it's seq. len)
    # - "constant" (divide all seq adv. by max seq. len)
    # - "microbatch" (divide all seq adv. by longest seq. in microbatch)
    normalize_mode: str = "constant"

    max_grad_norm: float = 1.0
    cliprange: float = 0.2
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    wandb_entity: str | None = "brandon-snider-stanford-university"
    wandb_project: str = "cs336-a5"
    # wandb_project: str | None = None
    torch_compile: bool = True
    log_step_interval: int = 1
    wandb_tags: list[str] = field(default_factory=lambda: ["grpo"])

    eval_before_training: bool = False
    eval_step_interval: int = 10
    eval_n_examples: int = 1024

    old_log_probs_batch_size: int = 8

    question_only: bool = False


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

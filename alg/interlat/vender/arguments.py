from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Interlat_preview/models/Qwen2.5-0.5B")
    trust_remote_code: bool = field(default=False)
    padding_side: str = field(default="right")
    prepended_length: int = field(default=800)
    prepended_hidden_state_path: Optional[str] = field(default=None)
    prepended_learnable: bool = field(default=False)
    prepend_position: str = field(default="first_human")
    plan_similarity_weight: float = field(default=1.0)
    random_contrast_weight: float = field(default=2.0)
    PE_fuse: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default="Interlat_preview/datasets/alfworld_sft.json")
    eval_data_path: Optional[str] = field(default=None)
    lazy_preprocess: bool = field(default=False)
    eval_ratio: float = field(default=0.01)
    hidden_data: str = field(default="pailitao_v100/alfworld_qwen05B_hidden")


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=3800)
    dataloader_num_workers: int = field(default=8)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_prefetch_factor: int = field(default=2)
    eval_steps: int = field(default=500)
    save_steps: int = field(default=100)
    logging_steps: int = field(default=20)
    save_total_limit: int = field(default=20)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    early_stopping_patience: int = field(default=100)
    early_stopping_threshold: float = field(default=0.0)

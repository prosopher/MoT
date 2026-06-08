from core.common import *
from core.context import Context


class InfiniteDataLoader:
    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


def build_training_dataloader(ctx: Context, tokenizer: PreTrainedTokenizerBase) -> InfiniteDataLoader:
    config = ctx.config
    dataset = OpenWebTextSequenceStream(
        tokenizer=tokenizer,
        sequence_length=config.total_tokens,
        split="train",
        shuffle=True,
        shuffle_buffer=config.shuffle_buffer,
        seed=config.seed,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0)
    return InfiniteDataLoader(dataloader)


def build_training_dataloaders_by_target(ctx: Context) -> Dict[str, InfiniteDataLoader]:
    target_node_ids = sorted({edge.tgt_id for edge in ctx.edges})
    return {
        node_id: build_training_dataloader(ctx, ctx.tp.get_tokenizer(node_id))
        for node_id in target_node_ids
    }


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(hidden)
        kv = self.context_norm(context)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        hidden = hidden + attn_out
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden


def blocks_to_partial_past_key_values(
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> PastKeyValues:
    batch_size, seq_len, num_layers, hidden_size = key_block.shape
    expected_hidden = num_heads * head_dim
    if hidden_size != expected_hidden:
        raise ValueError(f"Hidden mismatch: block has {hidden_size}, expected {expected_hidden}.")

    past_key_values = []
    for layer_idx in range(num_layers):
        key_layer = key_block[:, :, layer_idx, :]
        value_layer = value_block[:, :, layer_idx, :]
        key_layer = key_layer.view(batch_size, seq_len, num_heads, head_dim)
        value_layer = value_layer.view(batch_size, seq_len, num_heads, head_dim)
        key_layer = key_layer.permute(0, 2, 1, 3).contiguous()
        value_layer = value_layer.permute(0, 2, 1, 3).contiguous()
        past_key_values.append((key_layer, value_layer))
    return tuple(past_key_values)


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.step_id = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> None:
        self.step_id += 1
        if self.step_id <= self.warmup_steps:
            multiplier = self.step_id / self.warmup_steps
        else:
            progress = (self.step_id - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            param_group["lr"] = base_lr * multiplier

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def get_training_dtype(config) -> torch.dtype:
    if not hasattr(config, "dtype"):
        raise AttributeError("Training config must define a dtype field.")
    return get_torch_dtype(config.dtype)


def move_trainable_module_to_config_dtype(module: nn.Module, config) -> nn.Module:
    return module.to(device=config.device, dtype=get_training_dtype(config))


def build_models_and_tokenizers(
    config,
    nodes: List[Node],
) -> Tuple[Dict[str, PreTrainedModel], Dict[str, PreTrainedTokenizerBase]]:
    unique_tokenizers: Dict[str, PreTrainedTokenizerBase] = {}
    unique_models: Dict[str, PreTrainedModel] = {}

    tokenizers: Dict[str, PreTrainedTokenizerBase] = {}
    models: Dict[str, PreTrainedModel] = {}
    for node in nodes:
        if node.model_id not in unique_tokenizers:
            unique_tokenizers[node.model_id] = load_tokenizer(node.model_id)
        if node.model_id not in unique_models:
            unique_models[node.model_id] = load_frozen_model(node.model_id, device=config.device, dtype=config.dtype)
        tokenizers[node.id] = unique_tokenizers[node.model_id]
        models[node.id] = unique_models[node.model_id]

    return models, tokenizers


TRANSLATOR_CHECKPOINT_DIR_NAME = "translators"


def get_translator_checkpoint_dir(output_path: Union[str, Path]) -> Path:
    return Path(output_path) / TRANSLATOR_CHECKPOINT_DIR_NAME


def get_translator_checkpoint_filename(translator_id: str) -> str:
    from urllib.parse import quote

    return f"{quote(translator_id, safe='')}.pt"


def get_translator_checkpoint_path(output_path: Union[str, Path], translator_id: str) -> Path:
    return get_translator_checkpoint_dir(output_path) / get_translator_checkpoint_filename(translator_id)


def save_translator_checkpoints(output_path: Union[str, Path], translator_pool) -> Path:
    checkpoint_dir = get_translator_checkpoint_dir(output_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for translator_id, translator in translator_pool.translators.items():
        torch.save(translator.state_dict(), get_translator_checkpoint_path(output_path, translator_id))
    return checkpoint_dir


def load_translator_checkpoints(checkpoint_dir_path: Union[str, Path], translator_pool) -> None:
    for translator_id, translator in translator_pool.translators.items():
        checkpoint_path = get_translator_checkpoint_path(checkpoint_dir_path, translator_id)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Translator checkpoint not found: {checkpoint_path}")
        translator.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))

def get_train_config_path(output_path: Union[str, Path]) -> Path:
    return Path(output_path) / "train_config.json"


def get_train_log_path(output_path: Union[str, Path]) -> Path:
    return Path(output_path) / "train.log"


def get_train_checkpoint_path(output_path: Union[str, Path]) -> Path:
    return get_translator_checkpoint_dir(output_path)


def initialize_train_output_paths(config) -> None:
    output_path = config.output_path
    timestamp = config.timestamp

    if output_path is None:
        if not config.alg:
            return
        if timestamp is None:
            timestamp = build_timestamp_string()
            config.timestamp = timestamp
        output_path_obj = build_timestamped_output_path(
            alg=config.alg,
            timestamp=timestamp,
        )
    else:
        output_path_obj = Path(output_path)

    config.output_path = str(output_path_obj)

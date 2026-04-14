import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .channel_manager import Channel
from .common import (
    extract_past_key_values,
    read_json,
    split_prefix_and_suffix_for_exact_next_token_loss,
    write_json,
)
from .context import Context
from .topology import Edge
from .train_util import InfiniteDataLoader


@dataclass(frozen=True)
class ChannelProfileConfig:
    num_train_examples: int
    num_val_examples: int
    max_steps: int
    min_window_size: int
    translator_dim: int
    translator_heads: int
    translator_depth: int
    translator_mlp_ratio: int

    def __post_init__(self) -> None:
        if self.num_train_examples < 1:
            raise ValueError("num_train_examples must be >= 1")
        if self.num_val_examples < 1:
            raise ValueError("num_val_examples must be >= 1")
        if self.max_steps < 0:
            raise ValueError("max_steps must be >= 0")
        if self.min_window_size < 1:
            raise ValueError("min_window_size must be >= 1")
        if self.translator_heads < 1:
            raise ValueError("translator_heads must be >= 1")
        if self.translator_dim % self.translator_heads != 0:
            raise ValueError("translator_dim must be divisible by translator_heads")
        if self.translator_depth < 1:
            raise ValueError("translator_depth must be >= 1")
        if self.translator_mlp_ratio < 1:
            raise ValueError("translator_mlp_ratio must be >= 1")


@dataclass(frozen=True)
class ChannelProfileStep:
    removed_side: str
    window_size: int
    validation_loss: float
    improvement: float
    src_layer_start_idx: int
    src_layer_end_idx: int
    tgt_layer_start_idx: int
    tgt_layer_end_idx: int


@dataclass(frozen=True)
class ChannelProfileResult:
    edge_id: str
    selected_channels: List[Channel]
    best_validation_loss: float
    history: List[ChannelProfileStep]


@dataclass(frozen=True)
class ProxyValidationScore:
    translated_loss: float
    native_loss: float


def build_channel_profile_config_path(output_dir: Path) -> Path:
    return Path(output_dir) / "channel_profile.json"


class ChannelProfiler:
    def __init__(
        self,
        ctx: Context,
        profile_config: ChannelProfileConfig,
    ) -> None:
        self.ctx = ctx
        self.config = ctx.config
        self.profile_config = profile_config
        self.mm = ctx.mm

    def _get_logger(self) -> logging.Logger:
        train_logger = logging.getLogger(f"{self.config.alg}_train")
        if train_logger.handlers:
            return train_logger
        eval_logger = logging.getLogger(f"{self.config.alg}_eval")
        if eval_logger.handlers:
            return eval_logger
        return train_logger

    def profile_all_edges(self, edges: Optional[List[Edge]] = None) -> Dict[str, ChannelProfileResult]:
        logger = self._get_logger()
        results: Dict[str, ChannelProfileResult] = {}
        target_edges = self.ctx.edges if edges is None else edges
        for edge in target_edges:
            logger.info("[ChannelProfiler] profiling edge=%s (%s -> %s)", edge.id, edge.src_id, edge.tgt_id)
            result = self.profile_edge(edge)
            for channel in result.selected_channels:
                self.ctx.cm.add_channel(edge.id, channel.src_layer_idx, channel.dst_layer_idx)
            results[edge.id] = result
            logger.info(
                "[ChannelProfiler] %s selected src=L%d-L%d -> tgt=L%d-L%d (win=%d, val_loss=%.6f)",
                edge.id,
                result.selected_channels[0].src_layer_idx,
                result.selected_channels[-1].src_layer_idx,
                result.selected_channels[0].dst_layer_idx,
                result.selected_channels[-1].dst_layer_idx,
                len(result.selected_channels),
                result.best_validation_loss,
            )
        return results

    def profile_edge(self, edge: Edge) -> ChannelProfileResult:
        logger = self._get_logger()
        terminal_channels = self._build_terminal_channels(edge)
        logger.info(
            "[ChannelProfiler] %s initial terminal window src=L%d-L%d -> tgt=L%d-L%d (win=%d)",
            edge.id,
            terminal_channels[0].src_layer_idx,
            terminal_channels[-1].src_layer_idx,
            terminal_channels[0].dst_layer_idx,
            terminal_channels[-1].dst_layer_idx,
            len(terminal_channels),
        )
        train_bank = self._build_profile_bank(
            edge,
            num_examples=self.profile_config.num_train_examples,
            split="train",
            seed_offset=0,
        )
        val_bank = self._build_profile_bank(
            edge,
            num_examples=self.profile_config.num_val_examples,
            split="train",
            seed_offset=10_000,
        )
        logger.info(
            "[ChannelProfiler] %s profile bank train=%d val=%d proxy_steps=%d proxy_dim=%d heads=%d depth=%d",
            edge.id,
            len(train_bank),
            len(val_bank),
            self.profile_config.max_steps,
            self.profile_config.translator_dim,
            self.profile_config.translator_heads,
            self.profile_config.translator_depth,
        )

        probe_window_size = min(2, len(terminal_channels))
        probe_windows = [
            terminal_channels[idx : idx + probe_window_size]
            for idx in range(len(terminal_channels) - probe_window_size + 1)
        ]
        channel_scores: List[ProxyValidationScore] = []
        history: List[ChannelProfileStep] = []
        for channels in probe_windows:
            score = self._score_channels(edge, channels, train_bank, val_bank)
            channel_scores.append(score)
            history.append(
                self._build_history_step(
                    removed_side="probe",
                    channels=channels,
                    validation_loss=score.translated_loss,
                    improvement=0.0,
                )
            )

        translated_losses = [score.translated_loss for score in channel_scores]
        min_loss_idx = min(range(len(translated_losses)), key=translated_losses.__getitem__)
        lower_window_start_idx = self._find_lower_elbow_index(translated_losses, min_loss_idx)
        upper_window_start_idx = self._find_upper_elbow_index(translated_losses, min_loss_idx)
        lower_elbow_idx = lower_window_start_idx
        upper_elbow_idx = upper_window_start_idx + probe_window_size - 1
        lower_elbow_idx, upper_elbow_idx = self._enforce_min_window(
            lower_elbow_idx,
            upper_elbow_idx,
            anchor_idx=min_loss_idx,
            max_idx=len(terminal_channels) - 1,
        )
        selected_channels = terminal_channels[lower_elbow_idx : upper_elbow_idx + 1]
        selected_score = self._score_channels(edge, selected_channels, train_bank, val_bank)
        history.append(
            self._build_history_step(
                removed_side="selected",
                channels=selected_channels,
                validation_loss=selected_score.translated_loss,
                improvement=0.0,
            )
        )

        logger.info("")
        logger.info("[ChannelProfiler] %s Sliding-window validation profile (win=2)", edge.id)
        for idx, (channels, score) in enumerate(zip(probe_windows, channel_scores)):
            logger.info(
                "[ChannelProfiler] %s Probe[%02d] src=L%d-L%d tgt=L%d-L%d native=%.6f translated=%.6f",
                edge.id,
                idx,
                channels[0].src_layer_idx,
                channels[-1].src_layer_idx,
                channels[0].dst_layer_idx,
                channels[-1].dst_layer_idx,
                score.native_loss,
                score.translated_loss,
            )
        logger.info("")
        logger.info(
            "[ChannelProfiler] %s lowest validation probe idx=%d src=L%d-L%d tgt=L%d-L%d native=%.6f translated=%.6f",
            edge.id,
            min_loss_idx,
            probe_windows[min_loss_idx][0].src_layer_idx,
            probe_windows[min_loss_idx][-1].src_layer_idx,
            probe_windows[min_loss_idx][0].dst_layer_idx,
            probe_windows[min_loss_idx][-1].dst_layer_idx,
            channel_scores[min_loss_idx].native_loss,
            channel_scores[min_loss_idx].translated_loss,
        )
        logger.info(
            "[ChannelProfiler] %s lower elbow idx=%d src=L%d tgt=L%d translated=%.6f",
            edge.id,
            lower_elbow_idx,
            terminal_channels[lower_elbow_idx].src_layer_idx,
            terminal_channels[lower_elbow_idx].dst_layer_idx,
            channel_scores[lower_window_start_idx].translated_loss,
        )
        logger.info(
            "[ChannelProfiler] %s upper elbow idx=%d src=L%d tgt=L%d translated=%.6f",
            edge.id,
            upper_elbow_idx,
            terminal_channels[upper_elbow_idx].src_layer_idx,
            terminal_channels[upper_elbow_idx].dst_layer_idx,
            channel_scores[upper_window_start_idx].translated_loss,
        )
        logger.info(
            "[ChannelProfiler] %s selected window src=L%d-L%d -> tgt=L%d-L%d (win=%d) native=%.6f translated=%.6f",
            edge.id,
            selected_channels[0].src_layer_idx,
            selected_channels[-1].src_layer_idx,
            selected_channels[0].dst_layer_idx,
            selected_channels[-1].dst_layer_idx,
            len(selected_channels),
            selected_score.native_loss,
            selected_score.translated_loss,
        )

        return ChannelProfileResult(
            edge_id=edge.id,
            selected_channels=selected_channels,
            best_validation_loss=selected_score.translated_loss,
            history=history,
        )

    def _build_terminal_channels(self, edge: Edge) -> List[Channel]:
        src_spec = self.mm.get_model_spec(edge.src_id)
        tgt_spec = self.mm.get_model_spec(edge.tgt_id)
        window_size = min(src_spec.num_layers, tgt_spec.num_layers)
        src_start = src_spec.num_layers - window_size
        tgt_start = tgt_spec.num_layers - window_size
        return [
            Channel(src_layer_idx=src_start + offset, dst_layer_idx=tgt_start + offset)
            for offset in range(window_size)
        ]

    def _find_lower_elbow_index(self, losses: List[float], min_loss_idx: int) -> int:
        if len(losses) >= 2 and losses[0] < losses[1]:
            return 0
        if min_loss_idx <= 1:
            return min_loss_idx
        return self._find_segment_elbow_index(losses, start_idx=0, end_idx=min_loss_idx, fallback_idx=min_loss_idx)

    def _find_upper_elbow_index(self, losses: List[float], min_loss_idx: int) -> int:
        max_loss_idx = len(losses) - 1
        if len(losses) >= 2 and losses[max_loss_idx] < losses[max_loss_idx - 1]:
            return max_loss_idx
        if min_loss_idx >= max_loss_idx - 1:
            return min_loss_idx
        return self._find_segment_elbow_index(
            losses,
            start_idx=min_loss_idx,
            end_idx=max_loss_idx,
            fallback_idx=min_loss_idx,
        )

    def _find_segment_elbow_index(
        self,
        losses: List[float],
        *,
        start_idx: int,
        end_idx: int,
        fallback_idx: int,
    ) -> int:
        if end_idx - start_idx <= 1:
            return fallback_idx
        start_loss = losses[start_idx]
        end_loss = losses[end_idx]
        span = float(end_idx - start_idx)
        best_idx = fallback_idx
        best_distance = float('-inf')
        for idx in range(start_idx + 1, end_idx):
            ratio = (idx - start_idx) / span
            line_loss = start_loss + ratio * (end_loss - start_loss)
            distance = line_loss - losses[idx]
            if distance > best_distance:
                best_distance = distance
                best_idx = idx
        return best_idx if best_distance > 0.0 else fallback_idx

    def _enforce_min_window(
        self,
        lower_idx: int,
        upper_idx: int,
        *,
        anchor_idx: int,
        max_idx: int,
    ) -> tuple[int, int]:
        while upper_idx - lower_idx + 1 < self.profile_config.min_window_size:
            can_expand_lower = lower_idx > 0
            can_expand_upper = upper_idx < max_idx
            if not can_expand_lower and not can_expand_upper:
                break
            if not can_expand_lower:
                upper_idx += 1
                continue
            if not can_expand_upper:
                lower_idx -= 1
                continue
            lower_gap = anchor_idx - lower_idx
            upper_gap = upper_idx - anchor_idx
            if lower_gap <= upper_gap:
                lower_idx -= 1
            else:
                upper_idx += 1
        return lower_idx, upper_idx

    def _build_profile_bank(
        self,
        edge: Edge,
        *,
        num_examples: int,
        split: str,
        seed_offset: int,
    ) -> List[Dict[str, Any]]:
        from .common import OpenWebTextSequenceStream, compute_suffix_lm_loss
        from torch.utils.data import DataLoader

        dataset = OpenWebTextSequenceStream(
            tokenizer=self.ctx.tokenizer,
            sequence_length=self.config.total_tokens,
            split=split,
            shuffle=True,
            shuffle_buffer=self.config.shuffle_buffer,
            seed=self.config.seed + seed_offset,
        )
        loader = InfiniteDataLoader(DataLoader(dataset, batch_size=1, num_workers=0))
        source_model = self.mm.get_model(edge.src_id)
        target_model = self.mm.get_model(edge.tgt_id)

        bank: List[Dict[str, Any]] = []
        for _ in range(num_examples):
            input_ids = next(loader).to(self.config.device)
            prefix_cache_ids, lm_input_ids, lm_labels = split_prefix_and_suffix_for_exact_next_token_loss(
                input_ids=input_ids,
                prefix_tokens=self.config.prefix_tokens,
            )
            with torch.no_grad():
                native_target_past_key_values = extract_past_key_values(target_model, prefix_cache_ids)
                bank.append(
                    {
                        "prefix_cache_ids": prefix_cache_ids,
                        "lm_input_ids": lm_input_ids,
                        "lm_labels": lm_labels,
                        "source_past_key_values": extract_past_key_values(source_model, prefix_cache_ids),
                        "native_target_past_key_values": native_target_past_key_values,
                    }
                )
        return bank

    def _score_channels(
        self,
        edge: Edge,
        channels: List[Channel],
        train_bank: List[Dict[str, Any]],
        val_bank: List[Dict[str, Any]],
    ) -> ProxyValidationScore:
        proxy = self._fit_proxy_translator(edge, channels, train_bank)
        proxy.eval()
        translated_losses: List[float] = []
        native_losses: List[float] = []
        for sample in val_bank:
            with torch.no_grad():
                translated_losses.append(self._compute_translated_validation_loss(proxy, edge, channels, sample).item())
                native_losses.append(self._compute_native_validation_loss(edge, channels, sample).item())
        translated_loss = float(sum(translated_losses) / len(translated_losses))
        native_loss = float(sum(native_losses) / len(native_losses))
        return ProxyValidationScore(
            translated_loss=translated_loss,
            native_loss=native_loss,
        )

    def _fit_proxy_translator(
        self,
        edge: Edge,
        channels: List[Channel],
        train_bank: List[Dict[str, Any]],
    ):
        logger = self._get_logger()
        from mot.train import LayerWindowDirectionalTranslator

        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        logger.info("")
        logger.info(
            "[ChannelProfiler] %s train proxy win=%d src=L%d-L%d -> tgt=L%d-L%d steps=%d dim=%d heads=%d depth=%d",
            edge.id,
            len(channels),
            channels[0].src_layer_idx,
            channels[-1].src_layer_idx,
            channels[0].dst_layer_idx,
            channels[-1].dst_layer_idx,
            self.profile_config.max_steps,
            self.profile_config.translator_dim,
            self.profile_config.translator_heads,
            self.profile_config.translator_depth,
        )
        proxy = LayerWindowDirectionalTranslator(
            src_hidden_size=self.mm.get_model_spec(edge.src_id).hidden_size,
            tgt_hidden_size=self.mm.get_model_spec(edge.tgt_id).hidden_size,
            num_layers=len(channels),
            translator_dim=self.profile_config.translator_dim,
            translator_heads=self.profile_config.translator_heads,
            translator_depth=self.profile_config.translator_depth,
            mlp_ratio=self.profile_config.translator_mlp_ratio,
            variant="single",
            mot_num_translators=1,
            mot_top_k=1,
        ).to(self.config.device)
        if self.profile_config.max_steps < 1:
            return proxy

        optimizer = torch.optim.AdamW(
            proxy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        proxy.train()
        log_every = max(1, self.profile_config.max_steps // 4)
        last_loss = None
        for step_idx in range(self.profile_config.max_steps):
            sample = train_bank[step_idx % len(train_bank)]
            optimizer.zero_grad(set_to_none=True)
            loss = self._compute_proxy_loss(proxy, edge, channels, sample)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().item())
            if (
                step_idx == 0 or step_idx + 1 == self.profile_config.max_steps or (step_idx + 1) % log_every == 0
            ):
                logger.info(
                    "[ChannelProfiler] %s proxy train step=%d/%d loss=%.6f",
                    edge.id,
                    step_idx + 1,
                    self.profile_config.max_steps,
                    last_loss,
                )
        if last_loss is not None:
            logger.info(
                "[ChannelProfiler] %s proxy train done final_loss=%.6f",
                edge.id,
                last_loss,
            )
        return proxy

    def _compute_proxy_loss(
        self,
        proxy,
        edge: Edge,
        channels: List[Channel],
        sample: Dict[str, Any],
    ) -> torch.Tensor:
        from .common import compute_prefix_correction_and_suffix_lm_loss, past_key_values_to_blocks
        from mot.train import replay_target_prefill_with_injected_window

        selected_past = tuple(sample["source_past_key_values"][channel.src_layer_idx] for channel in channels)
        key_block, value_block = past_key_values_to_blocks(selected_past)
        translated_key, translated_value = proxy(key_block, value_block)
        tgt_spec = self.mm.get_model_spec(edge.tgt_id)
        target_model = self.mm.get_model(edge.tgt_id)
        mixed_target_past = replay_target_prefill_with_injected_window(
            target_model=target_model,
            prefix_input_ids=sample["prefix_cache_ids"],
            target_start_layer_idx=channels[0].dst_layer_idx,
            injected_key_block=translated_key,
            injected_value_block=translated_value,
            tgt_spec=tgt_spec,
        )
        return compute_prefix_correction_and_suffix_lm_loss(
            target_model=target_model,
            past_key_values=mixed_target_past,
            lm_input_ids=sample["lm_input_ids"],
            lm_labels=sample["lm_labels"],
            native_target_past_key_values=sample["native_target_past_key_values"],
            target_start_layer_idx=channels[0].dst_layer_idx,
        )

    def _compute_translated_validation_loss(
        self,
        proxy,
        edge: Edge,
        channels: List[Channel],
        sample: Dict[str, Any],
    ) -> torch.Tensor:
        from .common import compute_prefix_correction_and_suffix_lm_loss, past_key_values_to_blocks
        from mot.train import replay_target_prefill_with_injected_window

        selected_past = tuple(sample["source_past_key_values"][channel.src_layer_idx] for channel in channels)
        key_block, value_block = past_key_values_to_blocks(selected_past)
        translated_key, translated_value = proxy(key_block, value_block)
        tgt_spec = self.mm.get_model_spec(edge.tgt_id)
        target_model = self.mm.get_model(edge.tgt_id)
        mixed_target_past = replay_target_prefill_with_injected_window(
            target_model=target_model,
            prefix_input_ids=sample["prefix_cache_ids"],
            target_start_layer_idx=channels[0].dst_layer_idx,
            injected_key_block=translated_key,
            injected_value_block=translated_value,
            tgt_spec=tgt_spec,
        )
        return compute_prefix_correction_and_suffix_lm_loss(
            target_model=target_model,
            past_key_values=mixed_target_past,
            lm_input_ids=sample["lm_input_ids"],
            lm_labels=sample["lm_labels"],
            native_target_past_key_values=sample["native_target_past_key_values"],
            target_start_layer_idx=channels[0].dst_layer_idx,
        )

    def _compute_native_validation_loss(
        self,
        edge: Edge,
        channels: List[Channel],
        sample: Dict[str, Any],
    ) -> torch.Tensor:
        from .common import compute_prefix_correction_and_suffix_lm_loss

        return compute_prefix_correction_and_suffix_lm_loss(
            target_model=self.mm.get_model(edge.tgt_id),
            past_key_values=sample["native_target_past_key_values"],
            lm_input_ids=sample["lm_input_ids"],
            lm_labels=sample["lm_labels"],
            native_target_past_key_values=sample["native_target_past_key_values"],
            target_start_layer_idx=channels[0].dst_layer_idx,
        )

    def _build_history_step(
        self,
        *,
        removed_side: str,
        channels: List[Channel],
        validation_loss: float,
        improvement: float,
    ) -> ChannelProfileStep:
        return ChannelProfileStep(
            removed_side=removed_side,
            window_size=len(channels),
            validation_loss=validation_loss,
            improvement=improvement,
            src_layer_start_idx=channels[0].src_layer_idx,
            src_layer_end_idx=channels[-1].src_layer_idx,
            tgt_layer_start_idx=channels[0].dst_layer_idx,
            tgt_layer_end_idx=channels[-1].dst_layer_idx,
        )

    def _channels_from_history_step(self, step: ChannelProfileStep) -> List[Channel]:
        return [
            Channel(src_layer_idx=step.src_layer_start_idx + offset, dst_layer_idx=step.tgt_layer_start_idx + offset)
            for offset in range(step.window_size)
        ]


def save_channel_profile_config(output_dir: Path, profile_config: ChannelProfileConfig) -> Path:
    output_path = build_channel_profile_config_path(Path(output_dir))
    write_json(str(output_path), asdict(profile_config))
    return output_path


def load_channel_profile_config(config_path: Path) -> ChannelProfileConfig:
    return ChannelProfileConfig(**read_json(config_path))


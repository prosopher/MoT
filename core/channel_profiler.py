import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .channel_manager import Channel
from .common import (
    TokenIDs,
    extract_past_key_values,
    read_json,
    split_context_and_prompt_token_ids,
    write_json,
)
from .context import Context
from .topology import Edge, build_node_map
from .train_util import InfiniteDataLoader, get_training_dtype


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
    src_layer_indices: List[int]
    tgt_layer_indices: List[int]


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
        self.tp = ctx.tp
        self.cm = ctx.cm
        self.node_map = build_node_map(ctx.nodes)

    def profile_all_edges(self, edges: Optional[List[Edge]] = None) -> Dict[str, ChannelProfileResult]:
        results: Dict[str, ChannelProfileResult] = {}
        target_edges = self.ctx.edges if edges is None else edges
        for edge in target_edges:
            logging.info("[ChannelProfiler] profiling edge=%s (%s -> %s)", edge.id, edge.src_id, edge.tgt_id)
            result = self.profile_edge(edge)
            for channel in result.selected_channels:
                self.ctx.cm.add_channel(edge.id, channel.src_layer_idx, channel.dst_layer_idx)
            results[edge.id] = result
            logging.info(
                "[ChannelProfiler] %s selected pairs=%s (win=%d, val_loss=%.6f)",
                edge.id,
                self._format_channels(result.selected_channels),
                len(result.selected_channels),
                result.best_validation_loss,
            )
        return results

    def profile_edge(self, edge: Edge) -> ChannelProfileResult:
        if self.config.layer_alignment == "terminal":
            return self._profile_terminal_edge(edge)

        candidate_channels = self._build_candidate_channels(edge)
        logging.info(
            "[ChannelProfiler] %s initial %s candidates=%s (win=%d)",
            edge.id,
            self.config.layer_alignment,
            self._format_channels(candidate_channels),
            len(candidate_channels),
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
        logging.info(
            "[ChannelProfiler] %s profile bank train=%d val=%d proxy_steps=%d proxy_dim=%d heads=%d depth=%d",
            edge.id,
            len(train_bank),
            len(val_bank),
            self.profile_config.max_steps,
            self.profile_config.translator_dim,
            self.profile_config.translator_heads,
            self.profile_config.translator_depth,
        )

        min_model_layers = min(
            self.tp.get_model_spec(edge.src_id).num_layers,
            self.tp.get_model_spec(edge.tgt_id).num_layers,
        )
        min_probe_window_size = max(1, round(min_model_layers * self.config.min_window_size_ratio))
        probe_window_size = min(min_probe_window_size, len(candidate_channels))
        max_probe_window_size = max(
            probe_window_size,
            round(min_model_layers * self.config.max_window_size_ratio),
        )
        max_probe_window_size = min(max_probe_window_size, len(candidate_channels))
        probe_windows = [
            candidate_channels[idx : idx + probe_window_size]
            for idx in range(len(candidate_channels) - probe_window_size + 1)
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
        selected_channels, selected_score, expansion_history = self._expand_channels_greedily(
            edge=edge,
            candidate_channels=candidate_channels,
            probe_windows=probe_windows,
            probe_scores=channel_scores,
            best_probe_idx=min_loss_idx,
            max_probe_window_size=max_probe_window_size,
            train_bank=train_bank,
            val_bank=val_bank,
        )
        history.extend(expansion_history)

        logging.info("")
        logging.info("[ChannelProfiler] %s Sliding-window validation profile (win=%d)", edge.id, probe_window_size)
        for idx, (channels, score) in enumerate(zip(probe_windows, channel_scores)):
            logging.info(
                "[ChannelProfiler] %s Probe[%02d] pairs=%s native=%.6f translated=%.6f",
                edge.id,
                idx,
                self._format_channels(channels),
                score.native_loss,
                score.translated_loss,
            )
        logging.info("")
        logging.info(
            "[ChannelProfiler] %s lowest validation probe idx=%d pairs=%s native=%.6f translated=%.6f",
            edge.id,
            min_loss_idx,
            self._format_channels(probe_windows[min_loss_idx]),
            channel_scores[min_loss_idx].native_loss,
            channel_scores[min_loss_idx].translated_loss,
        )
        logging.info(
            "[ChannelProfiler] %s greedy seed pairs=%s (win=%d) native=%.6f translated=%.6f",
            edge.id,
            self._format_channels(probe_windows[min_loss_idx]),
            len(probe_windows[min_loss_idx]),
            channel_scores[min_loss_idx].native_loss,
            channel_scores[min_loss_idx].translated_loss,
        )
        for step in expansion_history:
            logging.info(
                "[ChannelProfiler] %s %s pairs=%s (win=%d) translated=%.6f improvement=%.6f",
                edge.id,
                step.removed_side,
                self._format_channels(self._channels_from_history_step(step)),
                step.window_size,
                step.validation_loss,
                step.improvement,
            )
        logging.info(
            "[ChannelProfiler] %s selected window pairs=%s (win=%d) native=%.6f translated=%.6f",
            edge.id,
            self._format_channels(selected_channels),
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

    def _profile_terminal_edge(self, edge: Edge) -> ChannelProfileResult:
        selected_channels = self._build_terminal_channels(edge)
        history = [
            self._build_history_step(
                removed_side="selected-terminal",
                channels=selected_channels,
                validation_loss=float("nan"),
                improvement=0.0,
            )
        ]
        logging.info(
            "[ChannelProfiler] %s terminal direct selected pairs=%s (win=%d, max_window_size_ratio=%.4f)",
            edge.id,
            self._format_channels(selected_channels),
            len(selected_channels),
            self.config.max_window_size_ratio,
        )
        return ChannelProfileResult(
            edge_id=edge.id,
            selected_channels=selected_channels,
            best_validation_loss=float("nan"),
            history=history,
        )

    def _build_terminal_channels(self, edge: Edge) -> List[Channel]:
        src_spec = self.tp.get_model_spec(edge.src_id)
        tgt_spec = self.tp.get_model_spec(edge.tgt_id)
        min_model_layers = min(src_spec.num_layers, tgt_spec.num_layers)
        window_size = max(1, round(min_model_layers * self.config.max_window_size_ratio))
        window_size = min(window_size, min_model_layers)
        src_start = src_spec.num_layers - window_size
        tgt_start = tgt_spec.num_layers - window_size
        return [
            Channel(src_layer_idx=src_start + offset, dst_layer_idx=tgt_start + offset)
            for offset in range(window_size)
        ]

    def _build_candidate_channels(self, edge: Edge) -> List[Channel]:
        if self.config.layer_alignment == "terminal":
            return self._build_terminal_channels(edge)
        if self.config.layer_alignment == "depth-ratio":
            return self._build_depth_ratio_channels(edge)
        raise ValueError(f"Unsupported layer_alignment for profiling: {self.config.layer_alignment}")

    def _build_depth_ratio_channels(self, edge: Edge) -> List[Channel]:
        src_spec = self.tp.get_model_spec(edge.src_id)
        tgt_spec = self.tp.get_model_spec(edge.tgt_id)
        num_pairs = min(src_spec.num_layers, tgt_spec.num_layers)
        src_layer_indices = self._build_depth_ratio_indices(src_spec.num_layers, num_pairs)
        tgt_layer_indices = self._build_depth_ratio_indices(tgt_spec.num_layers, num_pairs)
        return [
            Channel(src_layer_idx=src_layer_idx, dst_layer_idx=tgt_layer_idx)
            for src_layer_idx, tgt_layer_idx in zip(src_layer_indices, tgt_layer_indices)
        ]

    def _exclude_edge_probe_channels(self, edge: Edge, channels: List[Channel]) -> List[Channel]:
        src_spec = self.tp.get_model_spec(edge.src_id)
        tgt_spec = self.tp.get_model_spec(edge.tgt_id)
        if len(channels) <= 2 or min(src_spec.num_layers, tgt_spec.num_layers) <= 2:
            return channels

        filtered_channels = [
            channel
            for channel in channels
            if channel.src_layer_idx not in {0, src_spec.num_layers - 1}
            and channel.dst_layer_idx not in {0, tgt_spec.num_layers - 1}
        ]
        return filtered_channels or channels

    def _build_depth_ratio_indices(self, total_layers: int, num_pairs: int) -> List[int]:
        if num_pairs == 1:
            return [total_layers - 1]

        indices: List[int] = []
        for pos in range(num_pairs):
            ratio = pos / (num_pairs - 1)
            proposed_idx = int(round(ratio * (total_layers - 1)))
            remaining = num_pairs - pos - 1
            min_allowed = 0 if not indices else indices[-1] + 1
            max_allowed = total_layers - 1 - remaining
            indices.append(min(max(proposed_idx, min_allowed), max_allowed))
        return indices

    def _format_channels(self, channels: List[Channel]) -> str:
        return ", ".join(
            f"L{channel.src_layer_idx}->L{channel.dst_layer_idx}"
            for channel in channels
        )


    def _expand_channels_greedily(
        self,
        *,
        edge: Edge,
        candidate_channels: List[Channel],
        probe_windows: List[List[Channel]],
        probe_scores: List[ProxyValidationScore],
        best_probe_idx: int,
        max_probe_window_size: int,
        train_bank: List[Dict[str, Any]],
        val_bank: List[Dict[str, Any]],
    ) -> tuple[List[Channel], ProxyValidationScore, List[ChannelProfileStep]]:
        start_search_radius = 2
        score_cache: Dict[tuple[int, int], ProxyValidationScore] = {
            (idx, len(window)): score
            for idx, (window, score) in enumerate(zip(probe_windows, probe_scores))
        }

        current_start_idx = best_probe_idx
        current_window_size = len(probe_windows[best_probe_idx])
        current_channels = probe_windows[best_probe_idx]
        current_score = probe_scores[best_probe_idx]
        history: List[ChannelProfileStep] = []

        while current_window_size < max_probe_window_size:
            next_window_size = current_window_size + 1
            min_start_idx = max(0, current_start_idx - start_search_radius)
            max_start_idx = min(len(candidate_channels) - next_window_size, current_start_idx + start_search_radius)
            logging.info("")
            logging.info(
                "[ChannelProfiler] %s selected win=%d pairs=%s translated=%.6f",
                edge.id,
                current_window_size,
                self._format_channels(current_channels),
                current_score.translated_loss,
            )
            logging.info(
                "[ChannelProfiler] %s grow to win=%d search_start_idx=[%d,%d] (anchor_start=%d±%d)",
                edge.id,
                next_window_size,
                min_start_idx,
                max_start_idx,
                current_start_idx,
                start_search_radius,
            )
            local_candidates: List[tuple[int, List[Channel], ProxyValidationScore]] = []
            for start_idx in range(min_start_idx, max_start_idx + 1):
                channels = candidate_channels[start_idx : start_idx + next_window_size]
                window_key = (start_idx, next_window_size)
                if window_key in score_cache:
                    score = score_cache[window_key]
                else:
                    score = self._score_channels(edge, channels, train_bank, val_bank)
                    score_cache[window_key] = score
                local_candidates.append((start_idx, channels, score))

            if not local_candidates:
                break

            best_local_start_idx, best_local_channels, best_local_score = min(
                local_candidates,
                key=lambda candidate: (
                    candidate[2].translated_loss,
                    -candidate[1][0].src_layer_idx,
                    -candidate[1][0].dst_layer_idx,
                ),
            )

            current_start_idx = best_local_start_idx
            current_window_size = next_window_size
            current_channels = best_local_channels
            previous_score = current_score
            current_score = best_local_score
            history.append(
                self._build_history_step(
                    removed_side=f"expand-win-{current_window_size}",
                    channels=current_channels,
                    validation_loss=current_score.translated_loss,
                    improvement=previous_score.translated_loss - current_score.translated_loss,
                )
            )

        history.append(
            self._build_history_step(
                removed_side="selected",
                channels=current_channels,
                validation_loss=current_score.translated_loss,
                improvement=0.0,
            )
        )
        return current_channels, current_score, history

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

        source_model = self.tp.get_model(edge.src_id)
        target_model = self.tp.get_model(edge.tgt_id)

        def build_loader(model: Model) -> InfiniteDataLoader:
            dataset = OpenWebTextSequenceStream(
                tokenizer=model.tokenizer,
                sequence_length=self.config.total_tokens,
                split=split,
                shuffle=True,
                shuffle_buffer=self.config.shuffle_buffer,
                seed=self.config.seed + seed_offset,
            )

            def collate_token_ids(examples: List[torch.Tensor]) -> TokenIDs:
                return TokenIDs(
                    torch.stack([torch.as_tensor(example) for example in examples], dim=0),
                    model_id=model.id,
                )

            return InfiniteDataLoader(
                DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_token_ids)
            )

        source_loader = build_loader(source_model)
        target_loader = build_loader(target_model)

        bank: List[Dict[str, Any]] = []
        for _ in range(num_examples):
            source_token_ids = next(source_loader).to(self.config.device)
            target_token_ids = next(target_loader).to(self.config.device)
            source_context_token_ids = split_context_and_prompt_token_ids(
                token_ids=source_token_ids,
                context_tokens=self.config.prefix_tokens,
            )[0]
            target_context_token_ids, prompt_token_ids, label_token_ids = split_context_and_prompt_token_ids(
                token_ids=target_token_ids,
                context_tokens=self.config.prefix_tokens,
            )
            with torch.no_grad():
                native_target_past_key_values = extract_past_key_values(
                    target_model,
                    target_context_token_ids,
                )
                bank.append(
                    {
                        "target_context_token_ids": target_context_token_ids,
                        "prompt_token_ids": prompt_token_ids,
                        "label_token_ids": label_token_ids,
                        "source_past_key_values": extract_past_key_values(
                            source_model,
                            source_context_token_ids,
                        ),
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
        from alg.mot.train import LayerWindowDirectionalTranslator

        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        logging.info("")
        logging.info(
            "[ChannelProfiler] %s train proxy win=%d pairs=%s steps=%d",
            edge.id,
            len(channels),
            self._format_channels(channels),
            self.profile_config.max_steps,
        )
        src_spec = self.tp.get_model_spec(edge.src_id)
        tgt_spec = self.tp.get_model_spec(edge.tgt_id)
        proxy = LayerWindowDirectionalTranslator(
            src_hidden_size=src_spec.kv_hidden_size,
            tgt_hidden_size=tgt_spec.kv_hidden_size,
            num_layers=len(channels),
            translator_dim=self.profile_config.translator_dim,
            translator_heads=self.profile_config.translator_heads,
            translator_depth=self.profile_config.translator_depth,
            mlp_ratio=self.profile_config.translator_mlp_ratio,
            variant="single",
            mot_num_translators=1,
            mot_top_k=1,
        ).to(device=self.config.device, dtype=get_training_dtype(self.config))
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
                logging.info(
                    "[ChannelProfiler] %s proxy train step=%d/%d loss=%.6f",
                    edge.id,
                    step_idx + 1,
                    self.profile_config.max_steps,
                    last_loss,
                )
        if last_loss is not None:
            logging.info(
                "[ChannelProfiler] %s proxy train done final_loss=%.6f",
                edge.id,
                last_loss,
            )
        return proxy

    def _get_src_layer_indices(self, channels: List[Channel]) -> List[int]:
        src_layer_indices = [channel.src_layer_idx for channel in channels]
        self._validate_strictly_increasing_non_negative_indices(src_layer_indices, name="src_layer_indices")
        return src_layer_indices

    def _get_tgt_layer_indices(self, channels: List[Channel]) -> List[int]:
        tgt_layer_indices = [channel.dst_layer_idx for channel in channels]
        self._validate_strictly_increasing_non_negative_indices(tgt_layer_indices, name="target_layer_indices")
        return tgt_layer_indices

    def _validate_strictly_increasing_non_negative_indices(self, indices: List[int], *, name: str) -> None:
        if indices[0] < 0:
            raise ValueError(f"{name} must be >= 0, got {indices}")
        for prev_idx, next_idx in zip(indices, indices[1:]):
            if next_idx <= prev_idx:
                raise ValueError(f"{name} must be strictly increasing, got {indices}")

    def _compute_proxy_loss(
        self,
        proxy,
        edge: Edge,
        channels: List[Channel],
        sample: Dict[str, Any],
    ) -> torch.Tensor:
        from .common import compute_prefix_correction_and_suffix_lm_loss, past_key_values_to_blocks
        from alg.mot.train import replay_target_prefill_with_injected_window

        src_layer_indices = self._get_src_layer_indices(channels)
        target_layer_indices = self._get_tgt_layer_indices(channels)
        selected_past = tuple(sample["source_past_key_values"][layer_idx] for layer_idx in src_layer_indices)
        key_block, value_block = past_key_values_to_blocks(selected_past)
        translated_key, translated_value = proxy(key_block, value_block)
        tgt_spec = self.tp.get_model_spec(edge.tgt_id)
        target_model = self.tp.get_model(edge.tgt_id)
        mixed_target_past = replay_target_prefill_with_injected_window(
            target_model=target_model,
            context_token_ids=sample["target_context_token_ids"],
            target_layer_indices=target_layer_indices,
            injected_key_block=translated_key,
            injected_value_block=translated_value,
            tgt_spec=tgt_spec,
            target_model_id=self.node_map[edge.tgt_id].model_id,
        )
        return compute_prefix_correction_and_suffix_lm_loss(
            target_model=target_model,
            past_key_values=mixed_target_past,
            prompt_token_ids=sample["prompt_token_ids"],
            label_token_ids=sample["label_token_ids"],
            native_target_past_key_values=sample["native_target_past_key_values"],
            target_layer_indices=target_layer_indices,
        )

    def _compute_translated_validation_loss(
        self,
        proxy,
        edge: Edge,
        channels: List[Channel],
        sample: Dict[str, Any],
    ) -> torch.Tensor:
        from .common import compute_prefix_correction_and_suffix_lm_loss, past_key_values_to_blocks
        from alg.mot.train import replay_target_prefill_with_injected_window

        src_layer_indices = self._get_src_layer_indices(channels)
        target_layer_indices = self._get_tgt_layer_indices(channels)
        selected_past = tuple(sample["source_past_key_values"][layer_idx] for layer_idx in src_layer_indices)
        key_block, value_block = past_key_values_to_blocks(selected_past)
        translated_key, translated_value = proxy(key_block, value_block)
        tgt_spec = self.tp.get_model_spec(edge.tgt_id)
        target_model = self.tp.get_model(edge.tgt_id)
        mixed_target_past = replay_target_prefill_with_injected_window(
            target_model=target_model,
            context_token_ids=sample["target_context_token_ids"],
            target_layer_indices=target_layer_indices,
            injected_key_block=translated_key,
            injected_value_block=translated_value,
            tgt_spec=tgt_spec,
            target_model_id=self.node_map[edge.tgt_id].model_id,
        )
        return compute_prefix_correction_and_suffix_lm_loss(
            target_model=target_model,
            past_key_values=mixed_target_past,
            prompt_token_ids=sample["prompt_token_ids"],
            label_token_ids=sample["label_token_ids"],
            native_target_past_key_values=sample["native_target_past_key_values"],
            target_layer_indices=target_layer_indices,
        )

    def _compute_native_validation_loss(
        self,
        edge: Edge,
        channels: List[Channel],
        sample: Dict[str, Any],
    ) -> torch.Tensor:
        from .common import compute_prefix_correction_and_suffix_lm_loss

        target_layer_indices = self._get_tgt_layer_indices(channels)
        return compute_prefix_correction_and_suffix_lm_loss(
            target_model=self.tp.get_model(edge.tgt_id),
            past_key_values=sample["native_target_past_key_values"],
            prompt_token_ids=sample["prompt_token_ids"],
            label_token_ids=sample["label_token_ids"],
            native_target_past_key_values=sample["native_target_past_key_values"],
            target_layer_indices=target_layer_indices,
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
            src_layer_indices=[channel.src_layer_idx for channel in channels],
            tgt_layer_indices=[channel.dst_layer_idx for channel in channels],
        )

    def _channels_from_history_step(self, step: ChannelProfileStep) -> List[Channel]:
        return [
            Channel(src_layer_idx=src_layer_idx, dst_layer_idx=tgt_layer_idx)
            for src_layer_idx, tgt_layer_idx in zip(step.src_layer_indices, step.tgt_layer_indices)
        ]


def save_channel_profile_config(output_dir: Path, profile_config: ChannelProfileConfig) -> Path:
    output_path = build_channel_profile_config_path(Path(output_dir))
    write_json(str(output_path), asdict(profile_config))
    return output_path


def load_channel_profile_config(config_path: Path) -> ChannelProfileConfig:
    return ChannelProfileConfig(**read_json(config_path))


from __future__ import annotations

from typing import Any, Iterator

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class Model(nn.Module):
    """Wrapper that keeps a frozen LM and its tokenizer together."""

    id: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def __init__(
        self,
        id: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()
        self.id = id
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError as original_error:
            try:
                wrapped_model = super().__getattr__("model")
            except AttributeError:
                raise original_error
            return getattr(wrapped_model, name)

    @property
    def config(self) -> Any:
        return self.model.config

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return self.model.parameters(recurse=recurse)

from dataclasses import dataclass

from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class Model:
    id: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

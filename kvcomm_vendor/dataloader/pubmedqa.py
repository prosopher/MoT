from .base_evaluator import BaseEvaluator
from datasets import load_dataset


class PubMedQAEvaluator(BaseEvaluator):
    def __init__(self, n_samples=500):
        super().__init__()
        self.max_tokens = 8
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = n_samples
        self.data = self.load_data()
        self.name = "pubmedqa"

    def _normalize_context(self, context):
        if isinstance(context, str):
            return context.strip()
        if isinstance(context, list):
            return "\n".join(str(x).strip() for x in context if str(x).strip())
        if isinstance(context, dict):
            parts = []
            for key in ("contexts", "context", "text", "abstract", "passage", "sentences"):
                value = context.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
                elif isinstance(value, list):
                    parts.extend(str(x).strip() for x in value if str(x).strip())
            return "\n".join(parts).strip()
        return ""

    def load_data(self):
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
        dataset = self.random_sample(dataset)
        dataset = dataset.map(lambda x: {
            "prompt_A": self._normalize_context(x["context"]),
            "prompt_B": x["question"],
            "answer": str(x["final_decision"]).strip().lower(),
        })
        dataset = dataset.filter(lambda x: bool(x["prompt_A"]) and x["answer"] in {"yes", "no", "maybe"})
        return dataset

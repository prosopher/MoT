from .base_evaluator import BaseEvaluator
from datasets import load_dataset


class BoolQEvaluator(BaseEvaluator):
    def __init__(self, n_samples=500):
        super().__init__()
        self.max_tokens = 8
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = n_samples
        self.data = self.load_data()
        self.name = "boolq"

    def load_data(self):
        dataset = load_dataset("google/boolq")["validation"]
        dataset = self.random_sample(dataset)
        dataset = dataset.map(lambda x: {
            "prompt_A": x["passage"],
            "prompt_B": x["question"],
            "answer": "yes" if bool(x["answer"]) else "no",
        })
        return dataset

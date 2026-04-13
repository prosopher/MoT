from .base_evaluator import BaseEvaluator
from datasets import load_dataset


class SQuADEvaluator(BaseEvaluator):
    def __init__(self, n_samples=500):
        super().__init__()
        self.max_tokens = 32
        self.truncate_input = True
        self.multiple_answers = True
        self.n_samples = n_samples
        self.data = self.load_data()
        self.name = "squad"

    def load_data(self):
        dataset = load_dataset("rajpurkar/squad")["validation"]
        dataset = self.random_sample(dataset)
        dataset = dataset.map(lambda x: {
            "prompt_A": x["context"],
            "prompt_B": x["question"],
            "answers": [answer.strip() for answer in x["answers"]["text"] if isinstance(answer, str) and answer.strip()],
        })
        dataset = dataset.filter(lambda x: len(x["answers"]) > 0)
        return dataset

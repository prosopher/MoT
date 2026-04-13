from .base_evaluator import BaseEvaluator
from datasets import load_dataset


class NewsQAEvaluator(BaseEvaluator):
    def __init__(self, n_samples=500):
        super().__init__()
        self.max_tokens = 32
        self.truncate_input = True
        self.multiple_answers = True
        self.n_samples = n_samples
        self.data = self.load_data()
        self.name = "newsqa"

    @staticmethod
    def _normalize_answers(raw_answers):
        answers = []
        if isinstance(raw_answers, list):
            for raw_answer in raw_answers:
                if isinstance(raw_answer, str) and raw_answer.strip():
                    answers.append(raw_answer.strip())
                    continue
                if isinstance(raw_answer, dict):
                    text = raw_answer.get("text")
                    if isinstance(text, str) and text.strip():
                        answers.append(text.strip())
                    elif isinstance(text, list):
                        answers.extend(item.strip() for item in text if isinstance(item, str) and item.strip())
        elif isinstance(raw_answers, dict):
            text = raw_answers.get("text")
            if isinstance(text, str) and text.strip():
                answers.append(text.strip())
            elif isinstance(text, list):
                answers.extend(item.strip() for item in text if isinstance(item, str) and item.strip())
        return answers

    def load_data(self):
        dataset = load_dataset("gabrieltorresgamez/newsqa")["validation"]
        dataset = self.random_sample(dataset)

        rows = []
        for item in dataset:
            context = item.get("paragraph")
            if not isinstance(context, str) or not context.strip():
                continue
            raw_questions = item.get("questions")
            raw_answers = item.get("answers")
            if not isinstance(raw_questions, list) or not isinstance(raw_answers, list):
                continue
            for question, answers in zip(raw_questions, raw_answers):
                if not isinstance(question, str) or not question.strip():
                    continue
                answer_list = self._normalize_answers(answers)
                if not answer_list:
                    continue
                rows.append({
                    "prompt_A": context.strip(),
                    "prompt_B": question.strip(),
                    "answers": answer_list,
                })
                if self.n_samples is not None and len(rows) >= self.n_samples:
                    return rows
        return rows

"""Lightweight metric helpers for KVComm wrappers.

This version avoids network downloads during import so the HeteroCache wrapper can
run in offline environments. It preserves the public helpers used by the original
KVComm evaluators.
"""

import unicodedata

try:  # pragma: no cover
    import contractions  # type: ignore
except ImportError:  # pragma: no cover
    class _ContractionsModule:
        @staticmethod
        def fix(text):
            return text
    contractions = _ContractionsModule()

try:  # pragma: no cover
    import nltk  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore
    from nltk.corpus import wordnet  # type: ignore
    from nltk.tokenize import word_tokenize  # type: ignore
    _NLTK_AVAILABLE = True
except Exception:  # pragma: no cover
    nltk = None
    WordNetLemmatizer = None
    wordnet = None
    word_tokenize = None
    _NLTK_AVAILABLE = False

lemmatizer = WordNetLemmatizer() if _NLTK_AVAILABLE else None


def _remove_articles(text: str) -> str:
    words = [w for w in text.split() if w not in {"a", "an", "the"}]
    return " ".join(words)


def _remove_punctuation(text: str) -> str:
    return "".join(char for char in text if not unicodedata.category(char).startswith("P"))


def fix_answer(text):
    text = str(text).lower().strip()
    text = _remove_punctuation(text)
    text = _remove_articles(text)
    return " ".join(text.split())


def normalize_answer(text, lower=True):
    if isinstance(text, list):
        return [normalize_answer(item, lower=lower) for item in text]
    text = str(text)
    if lower:
        text = text.lower()
    text = _remove_punctuation(text)
    return fix_answer(" ".join(text.split()))


def remove_punctuation(text):
    try:
        return " ".join(_remove_punctuation(str(text).lower()).split())
    except Exception:
        return ""


def lemmatize_text(text):
    if not _NLTK_AVAILABLE:
        return text

    def get_wordnet_pos(word):
        try:
            tag = nltk.pos_tag([word])[0][1][0].upper()
        except Exception:
            return wordnet.NOUN
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }
        return tag_dict.get(tag, wordnet.NOUN)

    try:
        words = word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
        return " ".join(lemmatized_words)
    except Exception:
        return text


def f1_score_with_precision_recall(reference, candidate):
    reference = remove_punctuation(normalize_answer(reference))
    candidate = remove_punctuation(normalize_answer(candidate))
    words_reference = set(reference.split())
    words_candidate = set(candidate.split())
    tp = len(words_reference.intersection(words_candidate))
    fp = len(words_reference - words_candidate)
    fn = len(words_candidate - words_reference)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"f1": f1_score, "precision": precision, "recall": recall}


def calculate_f1_score_with_precision(str1, str2):
    str1 = fix_answer(contractions.fix(normalize_answer(str1)))
    str2 = fix_answer(contractions.fix(normalize_answer(str2)))
    words_str1 = set(str1.split())
    words_str2 = set(str2.split())
    tp = len(words_str1.intersection(words_str2))
    fp = len(words_str1 - words_str2)
    fn = len(words_str2 - words_str1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1_score, precision, recall

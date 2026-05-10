from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
import re
import signal
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import urllib.request
import zipfile


DOC2DIAL_DEFAULT_DATA_DIR = "./doc2dial_v1.0.1"
DOC2DIAL_DEFAULT_URL = "https://doc2dial.github.io/file/doc2dial_v1.0.1.zip"
DOC2DIAL_DOC_FILENAME = "doc2dial_doc.json"
DOC2DIAL_DIAL_TRAIN_FILENAME = "doc2dial_dial_train.json"
DOC2DIAL_DIAL_VALIDATION_FILENAME = "doc2dial_dial_validation.json"

# Backward-compatible constants for code that imported these names in the
# previous version. This loader uses only local raw JSON files.
DOC2DIAL_DIALOGUE_CONFIG = "dialogue_domain"
DOC2DIAL_DOCUMENT_CONFIG = "document_domain"
DOC2DIAL_HF_DATASET_PATH = "IBM/doc2dial"


@dataclass
class Doc2DialQAPair:
    """One multi-agent QA example extracted from a Doc2Dial dialogue.

    A Doc2Dial `dial_id` can contain many information-seeking exchanges. This
    dataclass represents one collapsed exchange: one or more consecutive user
    turns as the question, followed by one or more consecutive agent turns as the
    gold answer.
    """

    id: str
    dial_id: str
    doc_id: str
    domain: str
    question: str
    answers: List[str]
    context: str
    user_turn_ids: List[int]
    agent_turn_ids: List[int]
    reference_sp_ids: List[str]
    missing_reference_sp_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _split_to_dialogue_filename(split: str) -> str:
    split_norm = str(split or "validation").strip().lower()
    if split_norm in {"validation", "valid", "dev"}:
        return DOC2DIAL_DIAL_VALIDATION_FILENAME
    if split_norm == "train":
        return DOC2DIAL_DIAL_TRAIN_FILENAME
    if split_norm == "test":
        return "doc2dial_dial_test.json"
    return f"doc2dial_dial_{split_norm}.json"


def _find_file(root: Path, filename: str) -> Path:
    if root.is_file() and root.name == filename:
        return root
    if root.is_dir():
        direct = root / filename
        if direct.exists():
            return direct
        matches = sorted(root.rglob(filename))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {filename} under {root}")


def _has_required_doc2dial_files(root: Path) -> bool:
    if not root.exists():
        return False
    required = [DOC2DIAL_DOC_FILENAME, DOC2DIAL_DIAL_TRAIN_FILENAME, DOC2DIAL_DIAL_VALIDATION_FILENAME]
    for filename in required:
        try:
            _find_file(root, filename)
        except FileNotFoundError:
            return False
    return True


def ensure_doc2dial_data_dir(
    data_dir: str = DOC2DIAL_DEFAULT_DATA_DIR,
    *,
    url: str = DOC2DIAL_DEFAULT_URL,
) -> Path:
    """Return a local Doc2Dial v1.0.1 directory, downloading it if needed.

    The expected default layout is `./doc2dial_v1.0.1`. If that directory already
    contains `doc2dial_doc.json`, `doc2dial_dial_train.json`, and
    `doc2dial_dial_validation.json`, it is reused as-is. Otherwise, the official
    zip is downloaded and extracted into that directory.
    """

    root = Path(data_dir).expanduser()
    if _has_required_doc2dial_files(root):
        return root

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root.parent / f"{root.name}.zip"
    if not zip_path.exists():
        try:
            print(f"Downloading Doc2Dial v1.0.1 from {url} to {zip_path}")
            urllib.request.urlretrieve(url, zip_path)
        except Exception as exc:  # noqa: BLE001 - keep the error actionable for CLI users.
            raise RuntimeError(
                f"Doc2Dial data directory is missing required JSON files: {root}\n"
                f"Tried to download {url} but failed: {exc}\n"
                "Manually download doc2dial_v1.0.1.zip, extract it into "
                f"{root}, and make sure doc2dial_doc.json, doc2dial_dial_train.json, "
                "and doc2dial_dial_validation.json are present."
            ) from exc

    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Downloaded Doc2Dial archive is not a valid zip file: {zip_path}") from exc

    if not _has_required_doc2dial_files(root):
        raise FileNotFoundError(
            f"After extracting {zip_path} into {root}, required Doc2Dial JSON files were still not found."
        )
    return root


def _load_json_file(root: Path, filename: str) -> Dict[str, Any]:
    path = _find_file(root, filename)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a JSON object, got {type(data).__name__}")
    return data


def _iter_document_rows(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield normalized document rows from official doc2dial_doc.json."""

    doc_data = data.get("doc_data", data)

    if isinstance(doc_data, dict):
        for domain, docs_by_id in doc_data.items():
            if not isinstance(docs_by_id, dict):
                continue

            # Official shape: {domain: {doc_id: document}}
            if not any(k in docs_by_id for k in ("doc_id", "doc_text", "spans")):
                for doc_id, doc in docs_by_id.items():
                    if not isinstance(doc, dict):
                        continue
                    row = dict(doc)
                    row.setdefault("domain", str(domain))
                    row.setdefault("doc_id", str(doc_id))
                    yield row
                continue

            # Defensive shape: a single document object.
            row = dict(docs_by_id)
            row.setdefault("domain", str(row.get("domain", "")))
            row.setdefault("doc_id", str(row.get("doc_id", domain)))
            yield row


def _iter_dialogue_rows(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield normalized dialogue rows from official doc2dial_dial_*.json."""

    dial_data = data.get("dial_data", data)

    if isinstance(dial_data, dict):
        for domain, docs_by_id in dial_data.items():
            if isinstance(docs_by_id, list):
                for dialogue in docs_by_id:
                    if not isinstance(dialogue, dict):
                        continue
                    row = dict(dialogue)
                    row.setdefault("domain", str(domain))
                    row.setdefault("doc_id", str(row.get("doc_id", "")))
                    yield row
                continue

            if not isinstance(docs_by_id, dict):
                continue

            # Official shape: {domain: {doc_id: [dialogue, ...]}}
            if not any(k in docs_by_id for k in ("dial_id", "turns")):
                for doc_id, dialogues in docs_by_id.items():
                    for dialogue in _as_list(dialogues):
                        if not isinstance(dialogue, dict):
                            continue
                        row = dict(dialogue)
                        row.setdefault("domain", str(domain))
                        row.setdefault("doc_id", str(doc_id))
                        yield row
                continue

            # Defensive shape: a single dialogue object.
            row = dict(docs_by_id)
            row.setdefault("domain", str(row.get("domain", "")))
            row.setdefault("doc_id", str(row.get("doc_id", domain)))
            yield row
    elif isinstance(dial_data, list):
        for dialogue in dial_data:
            if isinstance(dialogue, dict):
                yield dict(dialogue)


def _span_id_from_reference(reference: Any) -> Optional[str]:
    if not isinstance(reference, dict):
        return None
    raw = reference.get("sp_id", reference.get("id_sp", reference.get("text_id")))
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _turn_id(turn: Dict[str, Any]) -> int:
    raw = turn.get("turn_id", 0)
    try:
        return int(raw)
    except Exception:
        return 0


def _turn_role(turn: Dict[str, Any]) -> str:
    return str(turn.get("role", "")).strip().lower()


def _turn_utterance(turn: Dict[str, Any]) -> str:
    return _clean_text(turn.get("utterance", ""))


def normalize_document_span_map(document_example: Dict[str, Any]) -> Dict[str, str]:
    """Return {sp_id: text_sp} for a Doc2Dial document JSON object."""

    spans = document_example.get("spans", {})
    span_map: Dict[str, str] = {}

    if isinstance(spans, dict):
        iterable: Iterable[Tuple[Any, Any]] = spans.items()
    elif isinstance(spans, list):
        iterable = ((None, item) for item in spans)
    else:
        iterable = []

    for key, raw_span in iterable:
        if not isinstance(raw_span, dict):
            continue
        raw_id = raw_span.get("sp_id", raw_span.get("id_sp", key))
        if raw_id is None:
            continue
        span_id = str(raw_id).strip()
        if not span_id:
            continue
        text_sp = _clean_text(raw_span.get("text_sp", ""))
        if text_sp:
            span_map[span_id] = text_sp

    return span_map


def _load_hf_dataset_with_compat(*, config_name: str, split: str):
    """Load a Hugging Face Doc2Dial config across old/new datasets versions."""

    from datasets import load_dataset  # type: ignore

    try:
        return load_dataset(DOC2DIAL_HF_DATASET_PATH, config_name, split=split, trust_remote_code=True)
    except TypeError:
        return load_dataset(DOC2DIAL_HF_DATASET_PATH, config_name, split=split)


def _dataset_to_dict_rows(dataset: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in dataset:
        if isinstance(item, dict):
            rows.append(dict(item))
    return rows


class _Doc2DialHFTimeout(TimeoutError):
    pass


def _run_with_timeout(callback, *, timeout_sec: int):
    if timeout_sec <= 0 or not hasattr(signal, "SIGALRM"):
        return callback()

    def _raise_timeout(signum, frame):
        del signum, frame
        raise _Doc2DialHFTimeout(f"Hugging Face Doc2Dial load timed out after {timeout_sec} seconds")

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(timeout_sec)
    try:
        return callback()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _try_load_doc2dial_from_huggingface(
    *,
    split: str = "validation",
    timeout_sec: int = 20,
) -> Optional[Tuple[Dict[Tuple[str, str], Dict[str, Any]], List[Dict[str, Any]]]]:
    """Try IBM/doc2dial first, then let the caller fall back to raw JSON.

    The Hugging Face dataset card exposes training/dev splits for the dialogue
    config and a training split for the document config, so documents are loaded
    from the document-domain training split and dialogues from the requested
    dialogue-domain split.
    """

    def _load_pair():
        dialogue_rows_inner = _dataset_to_dict_rows(
            _load_hf_dataset_with_compat(config_name=DOC2DIAL_DIALOGUE_CONFIG, split=split)
        )
        document_rows_inner = _dataset_to_dict_rows(
            _load_hf_dataset_with_compat(config_name=DOC2DIAL_DOCUMENT_CONFIG, split="train")
        )
        return document_rows_inner, dialogue_rows_inner

    try:
        document_rows, dialogue_rows = _run_with_timeout(_load_pair, timeout_sec=timeout_sec)
    except Exception:
        return None

    document_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for doc in document_rows:
        domain = str(doc.get("domain", "")).strip()
        doc_id = str(doc.get("doc_id", doc.get("title", ""))).strip()
        if not domain or not doc_id:
            continue
        document_index[(domain, doc_id)] = {
            "domain": domain,
            "doc_id": doc_id,
            "doc_text": _clean_text(doc.get("doc_text", "")),
            "spans": normalize_document_span_map(doc),
        }

    if not document_index or not dialogue_rows:
        return None
    return document_index, dialogue_rows


def build_doc2dial_document_index(
    *,
    data_dir: str = DOC2DIAL_DEFAULT_DATA_DIR,
    url: str = DOC2DIAL_DEFAULT_URL,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Parse doc2dial_doc.json and index documents by (domain, doc_id)."""

    root = ensure_doc2dial_data_dir(data_dir=data_dir, url=url)
    raw = _load_json_file(root, DOC2DIAL_DOC_FILENAME)
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for doc in _iter_document_rows(raw):
        domain = str(doc.get("domain", "")).strip()
        doc_id = str(doc.get("doc_id", doc.get("title", ""))).strip()
        if not domain or not doc_id:
            continue
        index[(domain, doc_id)] = {
            "domain": domain,
            "doc_id": doc_id,
            "doc_text": _clean_text(doc.get("doc_text", "")),
            "spans": normalize_document_span_map(doc),
        }
    return index


def _load_doc2dial_dialogues(
    *,
    data_dir: str = DOC2DIAL_DEFAULT_DATA_DIR,
    split: str = "validation",
    url: str = DOC2DIAL_DEFAULT_URL,
) -> List[Dict[str, Any]]:
    root = ensure_doc2dial_data_dir(data_dir=data_dir, url=url)
    raw = _load_json_file(root, _split_to_dialogue_filename(split))
    return list(_iter_dialogue_rows(raw))


def _collect_reference_span_ids(turns: Sequence[Dict[str, Any]]) -> List[str]:
    sp_ids: List[str] = []
    seen = set()
    for turn in turns:
        for reference in _as_list(turn.get("references", [])):
            sp_id = _span_id_from_reference(reference)
            if sp_id is None or sp_id in seen:
                continue
            sp_ids.append(sp_id)
            seen.add(sp_id)
    return sp_ids


def _build_context_from_spans(
    *,
    span_ids: Sequence[str],
    span_map: Dict[str, str],
    doc_text: str,
    context_max_chars: Optional[int] = None,
) -> Tuple[str, List[str]]:
    parts: List[str] = []
    missing: List[str] = []
    seen_text = set()

    for sp_id in span_ids:
        text = span_map.get(str(sp_id))
        if not text:
            missing.append(str(sp_id))
            continue
        if text in seen_text:
            continue
        parts.append(text)
        seen_text.add(text)

    # Some Doc2Dial turns are out-of-document/irrelevant and have no references.
    # Keep such examples runnable, but still prefer explicit span text whenever available.
    if not parts and doc_text:
        parts.append(doc_text)

    context = "\n".join(parts).strip()
    if context_max_chars is not None and context_max_chars > 0 and len(context) > context_max_chars:
        context = context[:context_max_chars].rstrip()
    return context, missing


def _select_context_turns(
    *,
    user_turns: Sequence[Dict[str, Any]],
    agent_turns: Sequence[Dict[str, Any]],
    context_reference_roles: str,
) -> List[Dict[str, Any]]:
    role_mode = context_reference_roles.lower().strip()
    if role_mode == "user":
        return list(user_turns)
    if role_mode == "agent":
        return list(agent_turns)
    if role_mode == "all":
        return list(user_turns) + list(agent_turns)
    raise ValueError("context_reference_roles must be one of: all, user, agent")


def extract_doc2dial_qa_pairs_from_dialogue(
    dialogue: Dict[str, Any],
    *,
    document_index: Dict[Tuple[str, str], Dict[str, Any]],
    context_reference_roles: str = "user",
    context_max_chars: Optional[int] = None,
) -> List[Doc2DialQAPair]:
    """Collapse consecutive user/agent turn blocks into multi-agent QA examples."""

    domain = str(dialogue.get("domain", "")).strip()
    doc_id = str(dialogue.get("doc_id", "")).strip()
    dial_id = str(dialogue.get("dial_id", "")).strip()
    doc = document_index.get((domain, doc_id), {})
    span_map: Dict[str, str] = doc.get("spans", {}) if isinstance(doc, dict) else {}
    doc_text: str = doc.get("doc_text", "") if isinstance(doc, dict) else ""

    raw_turns = dialogue.get("turns", [])
    turns = [turn for turn in _as_list(raw_turns) if isinstance(turn, dict)]
    turns.sort(key=_turn_id)

    examples: List[Doc2DialQAPair] = []
    i = 0
    qa_index = 0

    while i < len(turns):
        if _turn_role(turns[i]) != "user":
            i += 1
            continue

        user_turns: List[Dict[str, Any]] = []
        while i < len(turns) and _turn_role(turns[i]) == "user":
            if _turn_utterance(turns[i]):
                user_turns.append(turns[i])
            i += 1

        agent_turns: List[Dict[str, Any]] = []
        while i < len(turns) and _turn_role(turns[i]) == "agent":
            if _turn_utterance(turns[i]):
                agent_turns.append(turns[i])
            i += 1

        if not user_turns or not agent_turns:
            continue

        question = " ".join(_turn_utterance(turn) for turn in user_turns).strip()
        answer = " ".join(_turn_utterance(turn) for turn in agent_turns).strip()
        if not question or not answer:
            continue

        context_turns = _select_context_turns(
            user_turns=user_turns,
            agent_turns=agent_turns,
            context_reference_roles=context_reference_roles,
        )
        reference_sp_ids = _collect_reference_span_ids(context_turns)
        context, missing = _build_context_from_spans(
            span_ids=reference_sp_ids,
            span_map=span_map,
            doc_text=doc_text,
            context_max_chars=context_max_chars,
        )
        if not context:
            continue

        qa_index += 1
        examples.append(
            Doc2DialQAPair(
                id=f"{dial_id or domain + '_' + doc_id}#{qa_index}",
                dial_id=dial_id,
                doc_id=doc_id,
                domain=domain,
                question=question,
                answers=[answer],
                context=context,
                user_turn_ids=[_turn_id(turn) for turn in user_turns],
                agent_turn_ids=[_turn_id(turn) for turn in agent_turns],
                reference_sp_ids=reference_sp_ids,
                missing_reference_sp_ids=missing,
            )
        )

    return examples


def load_doc2dial_qa_pairs(
    *,
    data_dir: str = DOC2DIAL_DEFAULT_DATA_DIR,
    url: str = DOC2DIAL_DEFAULT_URL,
    split: str = "validation",
    max_examples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
    shuffle_buffer: int = 1024,
    domain_filter: Optional[str] = None,
    context_reference_roles: str = "user",
    context_max_chars: Optional[int] = None,
) -> List[Doc2DialQAPair]:
    """Load official Doc2Dial JSON files and convert them to QA examples.

    The conversion follows the requested multi-agent QA semantics: consecutive
    user turns are concatenated into one question, and the immediately following
    consecutive agent turns are concatenated into the gold answer.
    """

    root = Path(data_dir).expanduser()
    if _has_required_doc2dial_files(root):
        # Reuse an already prepared raw Doc2Dial directory immediately. This keeps
        # repeated experiments deterministic/offline and satisfies the "skip download
        # if ./doc2dial_v1.0.1 exists" requirement.
        document_index = build_doc2dial_document_index(data_dir=data_dir, url=url)
        dialogues = _load_doc2dial_dialogues(data_dir=data_dir, split=split, url=url)
    else:
        hf_loaded = _try_load_doc2dial_from_huggingface(split=split)
        if hf_loaded is None:
            document_index = build_doc2dial_document_index(data_dir=data_dir, url=url)
            dialogues = _load_doc2dial_dialogues(data_dir=data_dir, split=split, url=url)
        else:
            document_index, dialogues = hf_loaded

    iterable: Iterable[Dict[str, Any]] = dialogues
    if shuffle:
        rows = list(dialogues)
        rng = random.Random(seed)
        if shuffle_buffer and shuffle_buffer > 0 and shuffle_buffer < len(rows):
            # Keep the CLI meaning close to streaming shuffle, without an external loader.
            first = rows[:shuffle_buffer]
            rest = rows[shuffle_buffer:]
            rng.shuffle(first)
            rows = first + rest
        else:
            rng.shuffle(rows)
        iterable = rows

    examples: List[Doc2DialQAPair] = []
    domain_filter_norm = domain_filter.strip().lower() if isinstance(domain_filter, str) and domain_filter.strip() else None

    for dialogue in iterable:
        if not isinstance(dialogue, dict):
            continue
        if domain_filter_norm and str(dialogue.get("domain", "")).strip().lower() != domain_filter_norm:
            continue
        examples.extend(
            extract_doc2dial_qa_pairs_from_dialogue(
                dialogue,
                document_index=document_index,
                context_reference_roles=context_reference_roles,
                context_max_chars=context_max_chars,
            )
        )
        if max_examples is not None and max_examples > 0 and len(examples) >= max_examples:
            return examples[:max_examples]

    return examples


__all__ = [
    "DOC2DIAL_DEFAULT_DATA_DIR",
    "DOC2DIAL_DEFAULT_URL",
    "DOC2DIAL_DIALOGUE_CONFIG",
    "DOC2DIAL_DOCUMENT_CONFIG",
    "DOC2DIAL_HF_DATASET_PATH",
    "Doc2DialQAPair",
    "build_doc2dial_document_index",
    "ensure_doc2dial_data_dir",
    "extract_doc2dial_qa_pairs_from_dialogue",
    "load_doc2dial_qa_pairs",
    "normalize_document_span_map",
]

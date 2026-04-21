from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .common import read_json


GOLD_REFERENCE_LABELS = {"precondition", "solution"}
DEFAULT_DOC2DIAL_ROOT = Path("./doc2dial_data/doc2dial_v1.0.1")


@dataclass
class QATurn:
    qa_turn_idx: int
    user_turn_id: int
    agent_turn_id: int
    user_utterance: str
    gold_answer: str
    user_reference_spans: List[str]
    agent_reference_spans: List[str]


@dataclass
class DialogueEpisode:
    dial_id: str
    domain: str
    doc_id: str
    title: str
    gold_grounding_excerpt: str
    qa_turns: List[QATurn]


class Doc2DialEpisodeLoader:
    """
    Loads Doc2Dial dialogues from local JSON files under ./doc2dial_data/doc2dial_v1.0.1.

    Supported local files:
    - doc2dial_doc.json
    - doc2dial_dial_train.json
    - doc2dial_dial_validation.json
    - doc2dial_dial_dev.json (fallback for validation)
    """

    def __init__(
        self,
        *,
        split: str = "validation",
        min_qa_turns: int = 5,
        max_episodes: Optional[int] = None,
        dialogue_json_path: Optional[str] = None,
        document_json_path: Optional[str] = None,
        doc2dial_root: Optional[str] = None,
    ) -> None:
        self.split = split
        self.min_qa_turns = max(1, int(min_qa_turns))
        self.max_episodes = max_episodes
        self.dialogue_json_path = dialogue_json_path
        self.document_json_path = document_json_path
        self.doc2dial_root = doc2dial_root

    def load(self) -> List[DialogueEpisode]:
        self.dialogue_json_path, self.document_json_path = self._resolve_local_paths()
        return self._load_from_local_json()

    def _resolve_local_paths(self) -> Tuple[str, str]:
        if self.dialogue_json_path and self.document_json_path:
            dialogue_path = Path(self.dialogue_json_path).expanduser()
            document_path = Path(self.document_json_path).expanduser()
        else:
            root = Path(self.doc2dial_root).expanduser() if self.doc2dial_root else DEFAULT_DOC2DIAL_ROOT
            root = root.resolve(strict=False)

            dialogue_candidates = _candidate_dialogue_filenames(self.split)
            document_path = root / "doc2dial_doc.json"

            dialogue_path = None
            for filename in dialogue_candidates:
                candidate = root / filename
                if candidate.is_file():
                    dialogue_path = candidate
                    break

            if dialogue_path is None:
                raise FileNotFoundError(
                    "Doc2Dial dialogue JSON was not found under "
                    f"{root}. Tried: {', '.join(dialogue_candidates)}"
                )

        if not dialogue_path.is_file() or not document_path.is_file():
            raise FileNotFoundError(
                f"Doc2Dial local JSON not found: dialogue={dialogue_path}, document={document_path}"
            )
        return str(dialogue_path), str(document_path)

    def _load_from_local_json(self) -> List[DialogueEpisode]:
        if not self.dialogue_json_path or not self.document_json_path:
            raise ValueError(
                "Both dialogue_json_path and document_json_path are required for local Doc2Dial loading."
            )

        dialogue_payload = read_json(str(Path(self.dialogue_json_path)))
        document_payload = read_json(str(Path(self.document_json_path)))

        doc_rows = list(_iter_local_document_rows(document_payload))
        dialogue_rows = list(_iter_local_dialogue_rows(dialogue_payload))

        doc_map = {
            (str(doc["domain"]), str(doc["doc_id"])): doc
            for doc in doc_rows
        }

        episodes: List[DialogueEpisode] = []
        for dialogue in dialogue_rows:
            episode = self._build_episode_from_dialogue(dialogue, doc_map)
            if episode is None:
                continue
            episodes.append(episode)
            if self.max_episodes is not None and len(episodes) >= self.max_episodes:
                break
        return episodes

    def _build_episode_from_dialogue(
        self,
        dialogue: Dict[str, Any],
        doc_map: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Optional[DialogueEpisode]:
        domain = str(dialogue.get("domain", "")).strip()
        doc_id = str(dialogue.get("doc_id", "")).strip()
        dial_id = str(dialogue.get("dial_id", "")).strip()
        turns = dialogue.get("turns", [])

        if not domain or not doc_id or not dial_id or not isinstance(turns, list):
            return None

        document = doc_map.get((domain, doc_id))
        if document is None:
            return None

        qa_turns = _build_qa_turns(turns)
        if len(qa_turns) < self.min_qa_turns:
            return None

        excerpt = _build_gold_grounding_excerpt(dialogue=dialogue, document=document)
        if not excerpt:
            return None

        return DialogueEpisode(
            dial_id=dial_id,
            domain=domain,
            doc_id=doc_id,
            title=str(document.get("title", doc_id)).strip(),
            gold_grounding_excerpt=excerpt,
            qa_turns=qa_turns,
        )


def load_doc2dial_episodes(
    *,
    split: str = "validation",
    min_qa_turns: int = 5,
    max_episodes: Optional[int] = None,
    dialogue_json_path: Optional[str] = None,
    document_json_path: Optional[str] = None,
    doc2dial_root: Optional[str] = None,
) -> List[DialogueEpisode]:
    return Doc2DialEpisodeLoader(
        split=split,
        min_qa_turns=min_qa_turns,
        max_episodes=max_episodes,
        dialogue_json_path=dialogue_json_path,
        document_json_path=document_json_path,
        doc2dial_root=doc2dial_root,
    ).load()


def _build_qa_turns(turns: Sequence[Dict[str, Any]]) -> List[QATurn]:
    qa_turns: List[QATurn] = []
    qa_idx = 0

    for idx, turn in enumerate(turns[:-1]):
        if str(turn.get("role", "")).strip().lower() != "user":
            continue

        next_turn = turns[idx + 1]
        if str(next_turn.get("role", "")).strip().lower() != "agent":
            continue

        user_utterance = str(turn.get("utterance", "")).strip()
        gold_answer = str(next_turn.get("utterance", "")).strip()
        if not user_utterance or not gold_answer:
            continue

        qa_idx += 1
        qa_turns.append(
            QATurn(
                qa_turn_idx=qa_idx,
                user_turn_id=int(turn.get("turn_id", idx + 1)),
                agent_turn_id=int(next_turn.get("turn_id", idx + 2)),
                user_utterance=user_utterance,
                gold_answer=gold_answer,
                user_reference_spans=_extract_gold_reference_span_ids(
                    turn.get("references", turn.get("reference", []))
                ),
                agent_reference_spans=_extract_gold_reference_span_ids(
                    next_turn.get("references", next_turn.get("reference", []))
                ),
            )
        )

    return qa_turns


def _build_gold_grounding_excerpt(dialogue: Dict[str, Any], document: Dict[str, Any]) -> str:
    spans_payload = document.get("spans", {})
    if not isinstance(spans_payload, dict):
        return ""

    span_map: Dict[str, Dict[str, Any]] = {}
    for key, value in spans_payload.items():
        if not isinstance(value, dict):
            continue
        span_id = _normalize_span_id(value) or str(key)
        if not span_id:
            continue
        span_map[str(span_id)] = value

    seen = set()
    ordered_spans: List[Tuple[int, str]] = []
    for turn in dialogue.get("turns", []):
        refs = turn.get("references", turn.get("reference", []))
        for span_id in _extract_gold_reference_span_ids(refs):
            if span_id in seen:
                continue
            span = span_map.get(span_id)
            if span is None:
                continue
            text_sp = str(span.get("text_sp", "")).strip()
            if not text_sp:
                continue
            seen.add(span_id)
            start_sp = span.get("start_sp", 10 ** 12)
            try:
                sort_key = int(start_sp)
            except Exception:
                sort_key = 10 ** 12
            ordered_spans.append((sort_key, text_sp))

    ordered_spans.sort(key=lambda item: item[0])
    return "\n".join(text for _, text in ordered_spans).strip()


def _extract_gold_reference_span_ids(references: Any) -> List[str]:
    if not isinstance(references, list):
        return []

    span_ids: List[str] = []
    for reference in references:
        if not isinstance(reference, dict):
            continue

        label = str(reference.get("label", "")).strip().lower()
        if not label and "values" in reference:
            values = reference.get("values", [])
            if isinstance(values, list) and values:
                label = str(values[0]).strip().lower()
        if label and label not in GOLD_REFERENCE_LABELS:
            continue

        span_id = _normalize_span_id(reference)
        if span_id:
            span_ids.append(span_id)
    return span_ids


def _normalize_span_id(reference: Dict[str, Any]) -> Optional[str]:
    for key in ("sp_id", "id_sp", "keys"):
        value = reference.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def _iter_local_document_rows(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if "doc_data" in payload and isinstance(payload["doc_data"], dict):
        payload = payload["doc_data"]
    for domain, domain_docs in payload.items():
        if not isinstance(domain_docs, dict):
            continue
        for doc_id, doc in domain_docs.items():
            if not isinstance(doc, dict):
                continue
            row = dict(doc)
            row.setdefault("domain", domain)
            row.setdefault("doc_id", doc_id)
            yield row



def _iter_local_dialogue_rows(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if "dial_data" in payload and isinstance(payload["dial_data"], dict):
        payload = payload["dial_data"]
    for domain, domain_docs in payload.items():
        if not isinstance(domain_docs, dict):
            continue
        for doc_id, dialogues in domain_docs.items():
            if isinstance(dialogues, dict):
                iterable = dialogues.values()
            elif isinstance(dialogues, list):
                iterable = dialogues
            else:
                continue
            for dialogue in iterable:
                if not isinstance(dialogue, dict):
                    continue
                row = dict(dialogue)
                row.setdefault("domain", domain)
                row.setdefault("doc_id", doc_id)
                yield row



def _candidate_dialogue_filenames(split: str) -> List[str]:
    normalized = str(split).strip().lower()
    if normalized == "validation":
        return ["doc2dial_dial_validation.json", "doc2dial_dial_dev.json"]
    return [f"doc2dial_dial_{normalized}.json"]

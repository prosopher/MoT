from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Node:
    id: str
    model_id: str


@dataclass(frozen=True)
class Edge:
    id: str
    src_id: str
    dst_id: str


def index_to_node_id(index: int) -> str:
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")

    chars = []
    current = index
    while True:
        current, remainder = divmod(current, 26)
        chars.append(chr(ord("A") + remainder))
        if current == 0:
            break
        current -= 1
    return "".join(reversed(chars))


def parse_model_ids_csv(model_ids: str) -> List[str]:
    parsed = [item.strip() for item in str(model_ids).split(",") if item.strip()]
    if len(parsed) < 2:
        raise ValueError("model_ids must contain at least two comma-separated model ids.")
    return parsed


def build_nodes_from_model_ids(model_ids: str) -> List[Node]:
    return [
        Node(id=index_to_node_id(index), model_id=model_id)
        for index, model_id in enumerate(parse_model_ids_csv(model_ids))
    ]


def build_allowed_edge_ids(nodes: List[Node]) -> List[str]:
    edge_ids = []
    for src_node in nodes:
        for dst_node in nodes:
            if src_node.id == dst_node.id:
                continue
            edge_ids.append(f"{src_node.id}_to_{dst_node.id}")
    return edge_ids


def build_all_edges_from_nodes(nodes: List[Node]) -> List[Edge]:
    edges = []
    for src_node in nodes:
        for dst_node in nodes:
            if src_node.id == dst_node.id:
                continue
            edges.append(
                Edge(
                    id=f"{src_node.id}_to_{dst_node.id}",
                    src_id=src_node.id,
                    dst_id=dst_node.id,
                )
            )
    return edges


def parse_model_directions(model_directions: str, allowed_directions: Optional[Iterable[str]] = None) -> List[str]:
    parsed = [item.strip() for item in str(model_directions).split(",") if item.strip()]
    if not parsed:
        raise ValueError("model_directions must contain at least one edge.")

    allowed_set = None
    if allowed_directions is not None:
        allowed_set = set(allowed_directions)
        if not allowed_set:
            raise ValueError("allowed_directions must not be empty when provided.")

    deduped = []
    seen = set()
    for item in parsed:
        if allowed_set is not None and item not in allowed_set:
            raise ValueError(
                f"Unsupported model edge: {item}. "
                f"Allowed values are: {sorted(allowed_set)}"
            )
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def build_node_map(nodes: Iterable[Node]) -> Dict[str, Node]:
    return {node.id: node for node in nodes}


def build_edge_map(edges: Iterable[Edge]) -> Dict[str, Edge]:
    return {edge.id: edge for edge in edges}


def build_edges_from_nodes(nodes: List[Node], model_directions: str) -> List[Edge]:
    if str(model_directions).strip().lower() == "all":
        return build_all_edges_from_nodes(nodes)

    edge_ids = parse_model_directions(
        model_directions,
        allowed_directions=build_allowed_edge_ids(nodes),
    )

    return [
        Edge(
            id=edge_id,
            src_id=src_id,
            dst_id=dst_id,
        )
        for edge_id in edge_ids
        for src_id, dst_id in [edge_id.split("_to_", maxsplit=1)]
    ]


def build_nodes_and_edges(
    model_ids: str,
    model_directions: str,
) -> Tuple[List[Node], List[Edge]]:
    nodes = build_nodes_from_model_ids(model_ids)
    edges = build_edges_from_nodes(nodes, model_directions)
    return nodes, edges

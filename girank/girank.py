"""GIRank (Genetic Influence Rank) for a family tree."""

from __future__ import annotations

from typing import Iterable, Mapping

FAMILY_TREE: dict[str, list[str]] = {
    "Alice": ["Bob", "Charlie"],
    "Bob": ["David"],
    "Charlie": ["Eve", "Frank"],
    "David": ["George"],
    "Eve": ["Hannah", "Liam"],
    "Frank": ["Isaac"],
    "Zara": [],
}

EXPECTED_OUTPUT = """Genetic Influence Rank (ancestors):
Alice: 0.0530
Bob: 0.0755
Charlie: 0.0755
David: 0.1171
Eve: 0.0850
Frank: 0.0850
Zara: 0.0530
"""


def build_family_graph(
    family_tree: Mapping[str, Iterable[str]]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build children and parent lookup tables for a family tree."""
    all_nodes: set[str] = set(family_tree.keys())
    for children in family_tree.values():
        all_nodes.update(children)

    children_map: dict[str, set[str]] = {node: set() for node in all_nodes}
    parents_map: dict[str, set[str]] = {node: set() for node in all_nodes}

    for parent, children in family_tree.items():
        for child in children:
            children_map[parent].add(child)
            parents_map[child].add(parent)

    return children_map, parents_map


def girank_scores(
    children_map: Mapping[str, set[str]],
    *,
    damping: float = 0.85,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> dict[str, float]:
    """Compute Genetic Influence Rank scores using an iterative PageRank-style update."""
    if not 0 < damping < 1:
        raise ValueError("damping must be between 0 and 1")

    nodes = list(children_map.keys())
    node_count = len(nodes)
    if node_count == 0:
        return {}

    scores = {node: 1.0 / node_count for node in nodes}
    out_degree = {node: len(children_map[node]) for node in nodes}
    sink_nodes = {node for node, degree in out_degree.items() if degree == 0}

    base = (1 - damping) / node_count

    for _ in range(max_iterations):
        new_scores = {node: base for node in nodes}
        sink_score = sum(scores[node] for node in sink_nodes)
        sink_share = sink_score / node_count

        for parent in nodes:
            degree = out_degree[parent]
            if degree == 0:
                continue
            share = scores[parent] / degree
            for child in children_map[parent]:
                new_scores[child] += damping * share

        for node in nodes:
            new_scores[node] += damping * sink_share

        delta = max(abs(new_scores[node] - scores[node]) for node in nodes)
        scores = new_scores
        if delta < tolerance:
            break

    return scores


def ancestor_nodes(family_tree: Mapping[str, Iterable[str]]) -> set[str]:
    """Return nodes explicitly listed as parents in the input."""
    return set(family_tree.keys())


def main() -> None:
    """Run GIRank on the example tree and print ancestor scores."""
    children_map, _parents_map = build_family_graph(FAMILY_TREE)
    scores = girank_scores(children_map)
    ancestors = sorted(ancestor_nodes(FAMILY_TREE))

    print("Genetic Influence Rank (ancestors):")
    for ancestor in ancestors:
        print(f"{ancestor}: {scores[ancestor]:.4f}")


if __name__ == "__main__":
    main()

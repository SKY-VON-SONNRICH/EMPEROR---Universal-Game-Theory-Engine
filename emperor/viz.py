"""Tree visualization — inspect the search tree.

Exports: text_tree (terminal), json_tree (web UIs), mermaid_tree (docs).
"""

from __future__ import annotations
from .core import Node


def text_tree(root: Node, max_depth: int = 4, min_visits: int = 1) -> str:
    lines: list[str] = []
    _text_recurse(root, lines, "", True, max_depth, min_visits, 0)
    return "\n".join(lines)


def json_tree(root: Node, max_depth: int = 6, min_visits: int = 0) -> dict:
    return _json_node(root, max_depth, min_visits, 0)


def mermaid_tree(root: Node, max_depth: int = 3, min_visits: int = 2) -> str:
    lines = ["graph TD"]
    _counter = [0]
    _mermaid_recurse(root, "root", lines, _counter, max_depth, min_visits, 0)
    return "\n".join(lines)


# ── Internal ──────────────────────────────────────────────

def _q_dict(node: Node) -> dict[str, float]:
    if node.visits == 0:
        return {}
    return {p: round(v / node.visits, 3) for p, v in node.value_sum.items()}


def _label(node: Node) -> str:
    name = node.action.name if node.action else "root"
    q_str = ", ".join(f"{p}:{v:.0%}" for p, v in _q_dict(node).items())
    return f"{name} [n={node.visits} {q_str}]"


def _json_node(node: Node, max_d: int, min_v: int, depth: int) -> dict:
    result = {
        "action": node.action.name if node.action else "root",
        "description": (node.action.description if node.action else node.state.description[:120]),
        "visits": node.visits,
        "q": _q_dict(node),
        "prior": round(node.prior, 3),
    }
    if depth < max_d:
        kids = [
            _json_node(c, max_d, min_v, depth + 1)
            for c in sorted(node.children, key=lambda c: c.visits, reverse=True)
            if c.visits >= min_v
        ]
        if kids:
            result["children"] = kids
    return result


def _text_recurse(node, lines, prefix, is_last, max_d, min_v, depth):
    if depth > max_d or (depth > 0 and node.visits < min_v):
        return
    connector = "└── " if is_last else "├── "
    lines.append(f"{prefix}{connector if depth > 0 else ''}{_label(node)}")
    children = sorted(
        [c for c in node.children if c.visits >= min_v],
        key=lambda c: c.visits, reverse=True,
    )
    for i, child in enumerate(children):
        ext = "    " if is_last else "│   "
        _text_recurse(child, lines, prefix + (ext if depth > 0 else ""),
                       i == len(children) - 1, max_d, min_v, depth + 1)


def _mermaid_recurse(node, node_id, lines, counter, max_d, min_v, depth):
    if depth > max_d:
        return
    for child in node.children:
        if child.visits < min_v:
            continue
        counter[0] += 1
        cid = f"n{counter[0]}"
        name = child.action.name if child.action else "?"
        q = ", ".join(f"{p}:{v:.0%}" for p, v in _q_dict(child).items())
        pname = (node.action.name if node.action else "root").replace('"', "'")
        lines.append(f'    {node_id}["{pname} n={node.visits}"] --> {cid}["{name} n={child.visits} {q}"]')
        _mermaid_recurse(child, cid, lines, counter, max_d, min_v, depth + 1)

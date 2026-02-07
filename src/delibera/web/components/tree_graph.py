"""Convert ReplayedRun data to streamlit-agraph nodes and edges."""

from __future__ import annotations

from typing import Any

# Colors for node status
STATUS_COLORS = {
    "active": "#2196F3",  # blue
    "pruned": "#F44336",  # red
    "merged": "#4CAF50",  # green
}

# Sizes for node kinds
KIND_SIZES = {
    "root": 30,
    "option": 25,
    "plan": 20,
    "risk": 20,
    "merged": 35,
}

# Root gets a special color
ROOT_COLOR = "#607D8B"  # blue-grey
MERGED_COLOR = "#FF9800"  # gold


def build_agraph_data(
    replayed: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert ReplayedRun dict to agraph-compatible node and edge lists.

    Args:
        replayed: ReplayedRun serialized as dict.

    Returns:
        Tuple of (nodes, edges) as lists of dicts for streamlit-agraph.
    """
    nodes_data: list[dict[str, Any]] = []
    edges_data: list[dict[str, Any]] = []

    replayed_nodes = replayed.get("nodes", {})

    for nid, node in replayed_nodes.items():
        kind = node.get("kind", "option")
        status = node.get("status", "active")
        label = node.get("label", kind)
        depth = node.get("depth", 0)

        # Truncate label for display
        display_label = label[:35] + "..." if len(label) > 35 else label

        # Choose color
        if kind == "root":
            color = ROOT_COLOR
        elif kind == "merged":
            color = MERGED_COLOR
        else:
            color = STATUS_COLORS.get(status, "#2196F3")

        size = KIND_SIZES.get(kind, 20)

        # Build tooltip
        claim_summary = node.get("claim_summary")
        tooltip_parts = [
            f"Kind: {kind}",
            f"Status: {status}",
            f"Depth: {depth}",
        ]
        if claim_summary:
            tooltip_parts.append(
                f"Claims: {claim_summary.get('supported', 0)}S / "
                f"{claim_summary.get('weak', 0)}W / "
                f"{claim_summary.get('unsupported', 0)}U"
            )
        tooltip = "\n".join(tooltip_parts)

        nodes_data.append(
            {
                "id": nid,
                "label": display_label,
                "size": size,
                "color": color,
                "title": tooltip,
                "kind": kind,
                "status": status,
            }
        )

        # Add edge from parent
        parent_id = node.get("parent_id")
        if parent_id and parent_id in replayed_nodes:
            edges_data.append(
                {
                    "source": parent_id,
                    "target": nid,
                }
            )

    return nodes_data, edges_data


def build_graphviz_dot(replayed: dict[str, Any]) -> str:
    """Build a Graphviz DOT string as a fallback visualization.

    Args:
        replayed: ReplayedRun serialized as dict.

    Returns:
        DOT language string for graphviz rendering.
    """
    lines = ["digraph deliberation {"]
    lines.append("  rankdir=TB;")
    lines.append('  node [shape=box, style="rounded,filled", fontname="Helvetica"];')

    replayed_nodes = replayed.get("nodes", {})

    color_map = {
        "root": "#607D8B",
        "merged": "#FF9800",
    }
    status_color = {
        "active": "#2196F3",
        "pruned": "#F44336",
        "merged": "#4CAF50",
    }

    for nid, node in replayed_nodes.items():
        kind = node.get("kind", "option")
        status = node.get("status", "active")
        label = node.get("label", kind)
        display_label = label[:40].replace('"', '\\"')

        color = color_map.get(kind, status_color.get(status, "#2196F3"))

        lines.append(
            f'  "{nid}" [label="{display_label}", fillcolor="{color}", fontcolor="white"];'
        )

        parent_id = node.get("parent_id")
        if parent_id and parent_id in replayed_nodes:
            lines.append(f'  "{parent_id}" -> "{nid}";')

    lines.append("}")
    return "\n".join(lines)

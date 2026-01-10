"""Deliberation tree data structures."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from delibera.epistemics.ledger import Ledger


def _create_empty_ledger() -> Ledger:
    """Create an empty ledger for a new node."""
    from delibera.epistemics.ledger import Ledger

    return Ledger()


@dataclass
class Node:
    """A node in the deliberation tree.

    Each node represents a branch of deliberation (e.g., an option,
    hypothesis, plan, or risk).
    """

    node_id: str
    parent_id: str | None
    kind: Literal["root", "option", "merged"]
    depth: int
    status: Literal["active", "pruned", "merged"] = "active"
    artifact: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    ledger: Ledger = field(default_factory=_create_empty_ledger)


def _generate_id() -> str:
    """Generate a short, filesystem-safe ID."""
    return uuid.uuid4().hex[:8]


class DeliberationTree:
    """Manages the deliberation tree structure.

    The tree has exactly one root and supports operations for
    creating nodes, navigating the structure, and updating status.
    """

    def __init__(self) -> None:
        """Initialize an empty deliberation tree."""
        self._nodes: dict[str, Node] = {}
        self._root_id: str | None = None

    def create_root(self, question: str) -> Node:
        """Create the root node of the tree.

        Args:
            question: The initial question or problem statement.

        Returns:
            The created root node.

        Raises:
            ValueError: If a root already exists.
        """
        if self._root_id is not None:
            raise ValueError("Root node already exists")

        node_id = _generate_id()
        node = Node(
            node_id=node_id,
            parent_id=None,
            kind="root",
            depth=0,
            artifact={"question": question},
            label="root",
        )
        self._nodes[node_id] = node
        self._root_id = node_id
        return node

    def add_child(self, parent_id: str, label: str, kind: Literal["option", "merged"]) -> Node:
        """Add a child node to the tree.

        Args:
            parent_id: The ID of the parent node.
            label: A label for this branch (e.g., "Option A").
            kind: The kind of node ("option" or "merged").

        Returns:
            The created child node.

        Raises:
            KeyError: If the parent node does not exist.
        """
        parent = self._nodes[parent_id]
        node_id = _generate_id()
        node = Node(
            node_id=node_id,
            parent_id=parent_id,
            kind=kind,
            depth=parent.depth + 1,
            label=label,
        )
        self._nodes[node_id] = node
        return node

    def get_node(self, node_id: str) -> Node:
        """Get a node by ID.

        Args:
            node_id: The node's ID.

        Returns:
            The requested node.

        Raises:
            KeyError: If the node does not exist.
        """
        return self._nodes[node_id]

    def get_children(self, node_id: str) -> list[Node]:
        """Get all children of a node.

        Args:
            node_id: The parent node's ID.

        Returns:
            List of child nodes.
        """
        return [n for n in self._nodes.values() if n.parent_id == node_id]

    def get_root(self) -> Node:
        """Get the root node.

        Returns:
            The root node.

        Raises:
            ValueError: If no root exists.
        """
        if self._root_id is None:
            raise ValueError("No root node exists")
        return self._nodes[self._root_id]

    def set_status(self, node_id: str, status: Literal["active", "pruned", "merged"]) -> None:
        """Update a node's status.

        Args:
            node_id: The node's ID.
            status: The new status.

        Raises:
            KeyError: If the node does not exist.
        """
        self._nodes[node_id].status = status

    def update_artifact(self, node_id: str, artifact: dict[str, Any]) -> None:
        """Update a node's artifact.

        Args:
            node_id: The node's ID.
            artifact: The new artifact data.

        Raises:
            KeyError: If the node does not exist.
        """
        self._nodes[node_id].artifact = artifact

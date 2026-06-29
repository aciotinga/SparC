"""Deep-copy helpers for circuit DAGs."""

from __future__ import annotations

from sparc.nodes import (
    BernoulliInputNode,
    CategoricalInputNode,
    CircuitNode,
    DiscreteLogisticInputNode,
    IndicatorInputNode,
    LiteralInputNode,
    ProductNode,
    SumNode,
)


def clone_node(node: CircuitNode, memo: dict) -> CircuitNode:
    """Deep-copy a circuit DAG in memory, preserving id-based node sharing."""
    cached = memo.get(node.id)
    if cached is not None:
        return cached
    if isinstance(node, CategoricalInputNode):
        new = CategoricalInputNode(
            node.scope_as_list()[0],
            node.probabilities_list(),
            id=node.id,
        )
    elif isinstance(node, BernoulliInputNode):
        new = BernoulliInputNode(node.scope_as_list()[0], node.p(), id=node.id)
    elif isinstance(node, LiteralInputNode):
        new = LiteralInputNode(
            node.scope_as_list()[0], node.value_at(), id=node.id
        )
    elif isinstance(node, IndicatorInputNode):
        new = IndicatorInputNode(
            node.scope_as_list()[0],
            node.value_at(),
            node.num_categories(),
            id=node.id,
        )
    elif isinstance(node, DiscreteLogisticInputNode):
        new = DiscreteLogisticInputNode(
            node.scope_as_list()[0],
            node.mu_value(),
            node.s_value(),
            node.num_categories(),
            id=node.id,
        )
    elif isinstance(node, SumNode):
        children = [clone_node(c, memo) for c in node.children()]
        new = SumNode(children, node.parameters_list(), id=node.id)
    elif isinstance(node, ProductNode):
        children = [clone_node(c, memo) for c in node.children()]
        new = ProductNode(children, id=node.id)
    else:
        raise TypeError(
            f"clone() does not know how to copy {type(node).__name__}; "
            "custom leaf types should define their own cloning"
        )
    memo[node.id] = new
    return new

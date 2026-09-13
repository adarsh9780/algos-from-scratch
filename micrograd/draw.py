from graphviz import Digraph


def trace(root):
    """Recursively traces the computational graph starting at root,

    collecting all nodes and directed edges.
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    """Draws the computational graph using graphviz."""
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # Left to Right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))

        # 1. Create a rectangular node showing: label | data | grad
        dot.node(
            name=uid,
            label=f"{{ {n.label} | data {n.data:.4f} | grad {n.grad:.4f} }}",
            shape="record",
        )

        # 2. If this node was created by an operation (+, *), create a little op node
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            # Connect the op node to this result node
            dot.edge(uid + n._op, uid)

    # 3. Connect inputs to the operation node that consumed them
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

"""Visualization of the regex model, needs the optional dependencies."""
from __future__ import annotations

try:
    import networkx as nx
    from pyvis.network import Network
except ImportError as e:
    raise ValueError(
        "Please install visualization libraries 'networkx' and 'pyvis'.") from e

from regexmodel.model import RegexModel
from regexmodel.datastructure import RegexNode, Edge


def _create_network(graph: nx.DiGraph, edge: Edge, labels: list[str], prev_label_id: list[int]
                    ) -> list[int]:
    if edge.destination is None:
        return prev_label_id

    cur_i_label = len(labels)
    cur_node = edge.destination
    if isinstance(cur_node, RegexNode):
        cur_label = cur_node.regex
    else:
        cur_label = ""
    graph.add_node(cur_i_label, label=cur_label)
    for cur_id in prev_label_id:
        graph.add_edge(cur_id, cur_i_label)
    labels.append(cur_label)

    if isinstance(cur_node, RegexNode):
        return _create_network(graph, cur_node.next, labels, [cur_i_label])

    all_ends = []
    for cur_edge in cur_node.edges:  # type: ignore
        all_ends.extend(_create_network(graph, cur_edge, labels, [cur_i_label]))
    if cur_node.next.destination is None:
        return all_ends
    return _create_network(graph, cur_node.next, labels, all_ends)


def regex_model_to_pyvis(model: RegexModel, px_x: str = "1000px", px_y: str = "1000px",
                         notebook: bool = True) -> Network:
    """Convert the regex model to a PyVis network.

    Parameters
    ----------
    model:
        Regex model to be visualized.
    px_x:
        Number of pixels in the x-direction.
    px_y:
        Number of pixels in the y-direction.
    notebook:
        Whether to plot in a notebook.

    Returns
    -------
    network:
        A pyvis network corresponding to the regex model.
    """
    graph = nx.DiGraph()
    labels = ["s"]
    graph.add_node(0, label="start", group=2)
    # for edge in model.root_edges:
    _create_network(graph, model.regex_edge, labels, [0])

    net = Network(px_x, px_y, notebook=notebook, directed=True)
    net.from_nx(graph)
    return net

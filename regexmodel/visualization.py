"""Visualization of the regex model, needs the optional dependencies."""

try:
    import networkx as nx
    from pyvis.network import Network
except ImportError as e:
    raise ValueError(
        "Please install visualization libraries 'networkx' and 'pyvis'.") from e

from regexmodel.util import Dir
from regexmodel.model import RegexModel


def _create_network(graph, link, labels, node_start):
    cur_i_label = len(labels)
    if link.destination is None:
        graph.add_node(cur_i_label, label="X")
        labels.append("X")
        if link.direction == Dir.LEFT:
            graph.add_edge(cur_i_label, node_start, group=link.direction.value)
        else:
            graph.add_edge(node_start, cur_i_label, group=link.direction.value)
        return

    cur_node = link.destination
    if cur_node.regex is None:
        cur_label = "X"
    else:
        cur_label = cur_node.regex.regex
    graph.add_node(cur_i_label, label=cur_label)
    labels.append(cur_label)
    if link.direction == Dir.LEFT:
        graph.add_edge(cur_i_label, node_start, group=link.direction.value)
    else:
        graph.add_edge(node_start, cur_i_label, group=link.direction.value)

    for cur_link in cur_node.all_links:
        _create_network(graph, cur_link, labels, cur_i_label)


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
    for link in model.root_links:
        _create_network(graph, link, labels, 0)

    net = Network(px_x, px_y, notebook=notebook, directed=True)
    net.from_nx(graph)
    return net

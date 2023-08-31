try:
    import networkx as nx
    from pyvis.network import Network
except ImportError as e:
    raise ValueError(
        "Please install visualization libraries 'networkx' and 'pyvis'.") from e

from regexmodel.util import Dir


def _create_network(G, link, labels, node_start):
    cur_i_label = len(labels)
    if link.destination is None:
        G.add_node(cur_i_label, label="X")
        labels.append("X")
        if link.direction == Dir.LEFT:
            G.add_edge(cur_i_label, node_start, group=link.direction.value)
        else:
            G.add_edge(node_start, cur_i_label, group=link.direction.value)
        return

    cur_node = link.destination
    if cur_node.regex is None:
        cur_label = "X"
    else:
        cur_label = cur_node.regex.regex
    G.add_node(cur_i_label, label=cur_label)
    labels.append(cur_label)
    if link.direction == Dir.LEFT:
        G.add_edge(cur_i_label, node_start, group=link.direction.value)
    else:
        G.add_edge(node_start, cur_i_label, group=link.direction.value)

    for cur_link in cur_node.all_links:
        _create_network(G, cur_link, labels, cur_i_label)


def regex_model_to_pyvis(model, px_x="1000px", px_y="1000px", notebook=True):
    G = nx.DiGraph()
    labels = ["s"]
    G.add_node(0, label="start", group=2)
    for link in model.root_links:
        _create_network(G, link, labels, 0)

    net = Network(px_x, px_y, notebook=notebook, directed=True)
    net.from_nx(G)
    return net

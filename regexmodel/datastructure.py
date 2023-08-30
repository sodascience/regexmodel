"""Regex classes such as [0-9], [a-z], etc are defined in this module."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from regexmodel.util import Dir, sum_prob_log
from regexmodel.regexclass import regex_list_from_string, BaseRegex

LOG_LIKE_PER_CHAR = np.log(1e-3)
UNVIABLE_REGEX = -1000000


class Link():
    """Edge in the regex model.

    Links connect regexes and contain the weights that determine whether a
    link should be taken when a structured string is drawn.

    Parameters
    ----------
    count:
        Number of unstructured strings that are using this link.
    direction:
        Direction of the link.
    destination:
        Node to where the link points to.
    """

    def __init__(self, count: int, direction: Dir, destination: Optional[Node] = None):
        self.count = count
        self.direction = direction
        self.destination = destination

    @property
    def n_param(self):
        """Number of parameters used by this link."""
        if self.destination is None:
            return 1
        return self.destination.n_param + 1

    def check_zero_links(self):
        """Debug method."""
        if self.destination is None:
            return
        self.destination.check_zero_links()

    def log_likelihood(self, value: str) -> float:
        """Log likelihood computation over this link.

        Parameters
        ----------
        value:
            String to compute the log likelihood for.
        """
        if self.destination is None:
            # Check whether have reached the end in the expected way.
            if len(value) == 0:
                return 0
            return UNVIABLE_REGEX

        # If we are using the BOTH direction, we first loop over all starting points.
        if self.direction == Dir.BOTH:
            center_nodes = self.get_main_branch()
            weights = np.array([node.tot_weight(Dir.LEFT) for node in center_nodes])
            prob = weights/np.sum(weights)
            log_likes = np.array([node.log_likelihood(value, direction=self.direction)
                                  for node in center_nodes])
            return sum_prob_log(prob, log_likes)

        # Otherwise, just get the log likelihood for the destination.
        return self.destination.log_likelihood(value, direction=self.direction)

    def draw(self) -> str:
        """Draw a random string from this link."""
        if self.direction == Dir.BOTH:
            return self.draw_main()
        if self.destination is None:
            return ""
        return self.destination.draw(direction=self.direction)

    def draw_main(self) -> str:
        """Draw link using special method for drawing the main branch."""
        center_nodes = self.get_main_branch()
        if len(center_nodes) == 0:
            return ""
        weights = np.array([node.tot_weight(Dir.LEFT) for node in center_nodes])
        prob = weights/np.sum(weights)
        entry_node = np.random.choice(center_nodes, p=prob)  # type: ignore
        return entry_node.draw()

    def get_main_branch(self) -> list[Node]:
        """Create a list of nodes that form the main branch."""
        nodes = []
        next_node = self.destination
        while next_node is not None:
            if next_node is not None:
                nodes.append(next_node)
            else:
                break
            next_node = next_node.main_link.destination

        # Reverse the nodes if we are going backwards.
        if self.direction == Dir.LEFT:
            nodes.reverse()
        return nodes

    def serialize(self) -> dict:
        """Serialize the graph from the current link."""
        if self.destination is None:
            return {"weight": self.count}
        main_branch = self.get_main_branch()

        # Collect all the side branches.
        side_left: list[dict] = []
        side_right: list[dict] = []
        for i_node, node in enumerate(main_branch):
            for link in node.sub_links:
                if link.direction == Dir.LEFT:
                    cur_side = side_left
                else:
                    cur_side = side_right
                cur_side.append({
                    "i_branch": i_node,
                    "data": link.serialize()
                })
        return {
            "regex": "".join(node.regex.string for node in main_branch if node.regex is not None),
            "weights": [self.count] + [node.main_link.count for node in main_branch[:-1]],
            "side_branches_before": side_left,
            "side_branches_after": side_right,
        }

    def __repr__(self):
        if self.destination is not None:
            dest = str(self.destination.regex)
        else:
            dest = "NA"
        return f"Link <{self.count}, {self.direction}, {dest}>"

    @classmethod
    def deserialize(cls, param_dict: dict, direction=Dir.BOTH):
        """Deserialize from a parameter dictionary.

        See the serialize method on the structure for this dictionary.
        """
        # Check whether this is an end link.
        if "weight" in param_dict:
            return Link(param_dict["weight"], direction)

        # Create the main branch.
        weights = param_dict["weights"] + [0]
        regex_list = regex_list_from_string(param_dict["regex"])
        _main_link, node_list = Node.main_branch(regex_list, direction, weights)

        # Add side branches for Dir.LEFT
        for side_branch_dict in param_dict["side_branches_before"]:
            i_branch = side_branch_dict["i_branch"]
            data = side_branch_dict["data"]
            node_list[i_branch].sub_links.append(cls.deserialize(data, direction=Dir.LEFT))

        # Add side branches for Dir.RIGHT
        for side_branch_dict in param_dict["side_branches_after"]:
            i_branch = side_branch_dict["i_branch"]
            data = side_branch_dict["data"]
            node_list[i_branch].sub_links.append(cls.deserialize(data, direction=Dir.RIGHT))

        return cls(weights[0], direction, node_list[0])


class Node():
    """Vertex class for the regex model that contains the regex elements.

    Parameters
    ----------
    regex:
        Regex class for the current node.
    main_link:
        Link that follows the main branch.
    sub_links:
        All other links that originate from this node (any direction).
    """

    def __init__(self,
                 regex: Optional[BaseRegex],
                 main_link: Link,
                 sub_links: Optional[list[Link]] = None):
        self.regex = regex
        self.main_link = main_link
        self.sub_links = [] if sub_links is None else sub_links

    @property
    def all_links(self) -> list[Link]:
        """Get all the links including the main link."""
        return [self.main_link] + self.sub_links

    def draw_link(self, direction: Dir) -> Optional[Link]:
        """Get a random link according to their weights."""
        links = [link for link in self.all_links if link.direction == direction]
        if len(links) == 0:
            return None
        weights = np.array([link.count for link in links])
        chosen_link = np.random.choice(links, p=weights/np.sum(weights))  # type: ignore
        return chosen_link

    def tot_weight(self, direction: Optional[Dir]) -> int:
        """Get the total weight for all links in one direction.

        If direction is None, then get the weight for all links.
        """
        if direction is None:
            weights = [link.count for link in self.all_links]
        else:
            weights = [link.count for link in self.all_links if link.direction == direction]
        return int(np.sum(weights))

    def draw(self, direction: Dir = Dir.BOTH) -> str:
        """Draw a random string from the node."""
        # I think this is defunct, but should be tested.
        if self.regex is None and direction == Dir.BOTH:
            link = self.draw_link(direction)
            if link is not None:
                return link.draw()
            return ""

        # Draw a string before the regex node if left/both.
        cur_str = ""
        if direction in [Dir.LEFT, Dir.BOTH]:
            left_link = self.draw_link(direction=Dir.LEFT)
            if left_link is not None:
                cur_str += left_link.draw()

        # Draw the current regex itself.
        if self.regex is not None:
            cur_str += self.regex.draw()

        # Draw a string after the regex node if right/both.
        if direction in [Dir.RIGHT, Dir.BOTH]:
            right_link = self.draw_link(direction=Dir.RIGHT)
            if right_link is not None:
                cur_str += right_link.draw()
        return cur_str

    def check_zero_links(self):
        """Debug method."""
        for link in self.sub_links:
            assert link.count > 0
        for link in self.all_links:
            link.check_zero_links()

    @property
    def n_param(self) -> int:
        """Number of parameters of this node/downstream."""
        cur_param = 0
        for link in self.all_links:
            cur_param += link.n_param
        return cur_param

    def log_likelihood(self, value: str, direction: Dir) -> float:
        """Log likelihood calculation from this node."""
        if self.regex is None:
            return 0
        if len(value) == 0:
            return UNVIABLE_REGEX
        all_probs = []
        all_log_prob = []
        # Iterate over all ways the regex can be fit to the value.
        for res in self.regex.fit_value(value, direction):
            if direction == Dir.BOTH:
                pre_str, prob, post_str = res
                links_left = [link for link in self.all_links if link.direction == Dir.LEFT]

                # Only works if we have links on both sides.
                if len(links_left) == 0:
                    return UNVIABLE_REGEX
                links_right = [link for link in self.all_links if link.direction == Dir.RIGHT]
                loglike_left = _sum_links_loglike(pre_str, links_left)
                loglike_right = _sum_links_loglike(post_str, links_right)

                # Add log likelihood for both forward and backward probabilities.
                all_log_prob.append(loglike_left + loglike_right)
                all_probs.append(prob)
            else:
                pre_str, prob = res
                links = [link for link in self.all_links if link.direction == direction]
                log_like = _sum_links_loglike(pre_str, links)
                all_probs.append(prob)
                all_log_prob.append(log_like)
        if len(all_probs) == 0:
            return UNVIABLE_REGEX
        return sum_prob_log(all_probs, all_log_prob)

    @classmethod
    def main_branch(cls, regex_list: list[BaseRegex], direction: Dir,
                    weight: Union[int, list[int]] = 0
                    ) -> tuple[Link, list[Node]]:
        """Create a main branch out of a list of regex classes.

        Parameters
        ----------
        regex_list:
            List of regex classes to create a main branch out of.
        direction:
            Direction of the main branch.
        weight:
            The weights for the main branch. The length of this should be
            the length of the regex list + 1. If the weight is an integer,
            then the weights will be equal to that, except for the last link, which
            should always be 0.

        Returns
        -------
        root_link:
            Link that points to the new main branch.
        nodes_list:
            List of nodes that are interconnected and form the main branch.
        """
        main_dir = direction if direction != Dir.BOTH else Dir.RIGHT
        nodes = [Node(regex, Link(0, main_dir)) for regex in regex_list]
        if isinstance(weight, int):
            weight = [weight for _ in range(len(nodes) + 1)]
            if direction == Dir.LEFT:
                weight[0] = 0
            else:
                weight[-1] = 0
        assert len(weight) == len(regex_list) + 1
        if direction == Dir.LEFT:
            # link it up in reverse
            for i_node, node in enumerate(nodes[1:]):
                node.main_link = Link(weight[i_node+2], Dir.LEFT, nodes[i_node])
            return Link(weight[0], Dir.LEFT, nodes[-1]), nodes

        for i_node, node in enumerate(nodes[:-1]):
            node.main_link = Link(weight[i_node+1], main_dir, nodes[i_node+1])
        return Link(weight[0], direction, nodes[0]), nodes

    def __str__(self):
        return f"Node <{str(self.regex)}>"

    def __repr__(self):
        return f"Node <{str(self.regex)}>"


def _sum_links_loglike(value, links):
    """Get the log likelihood for a set of links (with weights)."""
    assert len(links) > 0
    probs = []
    log_likes = []
    for cur_link in links:
        log_likes.append(cur_link.log_likelihood(value))
        probs.append(cur_link.count)
    probs = np.array(probs)/np.sum(probs)
    return sum_prob_log(probs, log_likes)

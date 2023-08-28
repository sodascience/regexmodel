from __future__ import annotations

from typing import Optional, Union, Iterable

import numpy as np

from regexmodel.util import Dir, sum_prob_log
from regexmodel.regexclass import regex_list_from_string, BaseRegex

LOG_LIKE_PER_CHAR = np.log(1e-3)
UNVIABLE_REGEX = -1000000


class Link():
    def __init__(self, count, direction: Dir, destination=None):
        self.count = count
        self.direction = direction
        self.destination = destination
        assert self != self.destination

    @property
    def n_param(self):
        if self.destination is None:
            return 1
        return self.destination.n_param + 1

    def check_zero_links(self):
        if self.destination is None:
            return
        self.destination.check_zero_links()

    def log_likelihood(self, value: str) -> float:
        if self.destination is None:
            if len(value) == 0:
                return 0
            else:
                return UNVIABLE_REGEX
        if self.direction == Dir.BOTH:
            center_nodes = self.get_main_branch()
            weights = np.array([node.tot_weight(Dir.LEFT) for node in center_nodes])
            prob = weights/np.sum(weights)
            log_likes = np.array([node.log_likelihood(value, direction=self.direction)
                                  for node in center_nodes])
            return sum_prob_log(prob, log_likes)
        return self.destination.log_likelihood(value, direction=self.direction)

    def draw(self):
        if self.direction == Dir.BOTH:
            return self.draw_main()
        if self.destination is None:
            return ""
        return self.destination.draw(direction=self.direction)

    def draw_main(self):
        center_nodes = self.get_main_branch()
        if len(center_nodes) == 0:
            return ""
        weights = np.array([node.tot_weight(Dir.LEFT) for node in center_nodes])
        prob = weights/np.sum(weights)
        entry_node = np.random.choice(center_nodes, p=prob)
        return entry_node.draw()

    def get_main_branch(self):
        # cur_dir = self.direction if self.direction != Dir.BOTH else Dir.RIGHT
        nodes = []
        next_node = self.destination
        while next_node is not None:
            if next_node is not None:
                nodes.append(next_node)
            else:
                break
            next_node = next_node.main_link.destination

        if self.direction == Dir.LEFT:
            nodes.reverse()
        return nodes

    def _param_dict(self):
        if self.destination is None:
            return {"weight": self.count}
        main_branch = self.get_main_branch()

        side_left = []
        side_right = []
        for i_node, node in enumerate(main_branch):
            for link in node.sub_links:
                if link.direction == Dir.LEFT:
                    cur_side = side_left
                else:
                    cur_side = side_right
                cur_side.append({
                    "i_branch": i_node,
                    "data": link._param_dict()
                })
        return {
            "regex": "".join(node.regex.string for node in main_branch),
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
    def from_dict(cls, param_dict, direction=Dir.BOTH):
        if "weight" in param_dict:
            return Link(param_dict["weight"], direction)
        weights = param_dict["weights"] + [0]
        regex_list = regex_list_from_string(param_dict["regex"])
        main_link, node_list = Node.main_branch(regex_list, direction, weights)
        for side_branch_dict in param_dict["side_branches_before"]:
            i_branch = side_branch_dict["i_branch"]
            data = side_branch_dict["data"]
            node_list[i_branch].sub_links.append(cls.from_dict(data, direction=Dir.LEFT))
        for side_branch_dict in param_dict["side_branches_after"]:
            i_branch = side_branch_dict["i_branch"]
            data = side_branch_dict["data"]
            node_list[i_branch].sub_links.append(cls.from_dict(data, direction=Dir.RIGHT))

        return cls(weights[0], direction, node_list[0])


class Node():
    def __init__(self,
                 regex: Optional[BaseRegex],
                 main_link: Link,
                 sub_links: Optional[list[Link]] = None):
        self.regex = regex
        self.main_link = main_link
        self.sub_links = [] if sub_links is None else sub_links
        assert isinstance(regex, BaseRegex) or regex is None

    @property
    def all_links(self):
        return [self.main_link] + self.sub_links

    def draw_link(self, direction):
        links = [link for link in self.all_links if link.direction == direction]
        if len(links) == 0:
            return None
        weights = np.array([link.count for link in links])
        chosen_link = np.random.choice(links, p=weights/np.sum(weights))
        return chosen_link

    def tot_weight(self, direction: Optional[Dir]):
        if direction is None:
            weights = [link.count for link in self.all_links]
        else:
            weights = [link.count for link in self.all_links if link.direction == direction]
        return int(np.sum(weights))

    def draw(self, direction: Dir = Dir.BOTH) -> str:
        if self.regex is None and direction == Dir.BOTH:
            link = self.draw_link(direction)
            return link.draw()

        cur_str = ""
        if direction in [Dir.LEFT, Dir.BOTH]:
            left_link = self.draw_link(direction=Dir.LEFT)
            if left_link is not None:
                cur_str += left_link.draw()

        if self.regex is not None:
            cur_str += self.regex.draw()

        if direction in [Dir.RIGHT, Dir.BOTH]:
            right_link = self.draw_link(direction=Dir.RIGHT)
            if right_link is not None:
                cur_str += right_link.draw()
        return cur_str

    def check_zero_links(self):
        for link in self.sub_links:
            assert link.count > 0
        for link in self.all_links:
            link.check_zero_links()

    @property
    def n_param(self):
        cur_param = 0
        for link in self.all_links:
            cur_param += link.n_param
        return cur_param

    def log_likelihood(self, value: str, direction: Dir) -> float:
        if self.regex is None:
            return 0
        if len(value) == 0:
            return UNVIABLE_REGEX
        all_probs = []
        all_log_prob = []
        for res in self.regex.fit_value(value, direction):
            if direction == Dir.BOTH:
                pre_str, prob, post_str = res
                links_left = [link for link in self.all_links if link.direction == Dir.LEFT]
                if len(links_left) == 0:
                    return UNVIABLE_REGEX
                links_right = [link for link in self.all_links if link.direction == Dir.RIGHT]
                loglike_left = _sum_links_loglike(pre_str, links_left)
                loglike_right = _sum_links_loglike(post_str, links_right)
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
        else:
            for i_node, node in enumerate(nodes[:-1]):
                node.main_link = Link(weight[i_node+1], main_dir, nodes[i_node+1])
            return Link(weight[0], direction, nodes[0]), nodes

    def __str__(self):
        return f"Node <{str(self.regex)}>"

    def __repr__(self):
        return f"Node <{str(self.regex)}>"


def _sum_links_loglike(value, links):
    assert len(links) > 0
    probs = []
    log_likes = []
    for cur_link in links:
        log_likes.append(cur_link.log_likelihood(value))
        # assert cur_link.count > 0
        probs.append(cur_link.count)
    probs = np.array(probs)/np.sum(probs)
    return sum_prob_log(probs, log_likes)

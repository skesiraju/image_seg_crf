#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Thu Oct 27 13:24:36 2016
# @authors: Santosh, Srikanth

"""
Belief Propagation
1. Sum of products (forward, backward)
2. Max sum
"""

from collections import OrderedDict
from pprint import pprint
import sys
import argparse
import numpy as np
from numpy import asarray as asa

def format_probs(probs):
    """ Format 1D probs into 2D """

    p_new = OrderedDict()
    for key, val in probs.items():
        p_new[key] = asa([val, 1. - val]).reshape(-1, 1)
    return p_new


def normalize_msg(msg):
    """ Normalize the message """
    return msg / msg.sum()

class BeliefPropagation:
    """ Belief Propagation for tree """

    def __init__(self):

        self.tree = None

    def get_tree(self):
        """ Load the patches into a tree """

        pair_pot_12 = asa([[1, 2], [2, 4]])
        pair_pot_23 = asa([[2, 2], [1, 4]])
        tree = OrderedDict({1: {2: [3]}})
        probs = OrderedDict({1: asa([0.5, 0.5]).reshape(-1, 1),
                             2: asa([0.5, 0.5]).reshape(-1, 1),
                             3: asa([0.5, 0.5]).reshape(-1, 1)})

        tree = OrderedDict({1: {2: [5, 6], 3: [7], 4: [8, 9]}})
        probs = OrderedDict({1: 0.7, 2: 0.5, 3: 0.6, 4: 0.5, 5: 0.5,
                             6: 0.3, 7: 0.4, 8: 0.1, 9: 0.8})
        probs = format_probs(probs)
        pair_pot_12 = asa([[4, 1], [1, 4]])
        pair_pot_23 = asa([[2, 1], [1, 2]])

        pair_pot = {1: pair_pot_12, 2: pair_pot_23}

        return tree, probs, pair_pot

    def construct_factor_graph(self, tree):
        """ Construct factor graph from a given tree """

        factor_graph = {}
        for parent, children in tree.items():
            if isinstance(children, list):
                for child in children:
                    try:
                        factor_graph[parent].append((parent, child))
                    except KeyError:
                        factor_graph[parent] = [(parent, child)]

            elif isinstance(children, dict):
                for child in children.keys():
                    try:
                        factor_graph[parent].append((parent, child))
                    except KeyError:
                        factor_graph[parent] = [(parent, child)]

                sub_factor_graph = self.construct_factor_graph(children)
                for par, chl in sub_factor_graph.items():
                    try:
                        factor_graph[par].append(chl)
                    except KeyError:
                        factor_graph[par] = chl
            else:
                try:
                    factor_graph[parent].append((parent, children))
                except KeyError:
                    factor_graph[parent] = [(parent, children)]

        return factor_graph

    def leaf_to_root(self, tree, fgraph, prob, pair_pot):
        """ Send messages from leaf to root to compute marginal of root """

        # For each leaf to parent node, send the message through the factor b/w them
        # Step 1: Send msg to factor node
        # Step 2: Dot product with factor matrix and msg received
        # Parent receives multiple messages, one from each child (leaf in our case)

        msgs = {}  # messages that are received at node from child nodes

        for rnode, stree in tree.items():
            if isinstance(stree, dict):
                btm_msgs = self.leaf_to_root(stree, fgraph, prob, pair_pot)
                for bnode, msg_val in btm_msgs.items():
                    try:
                        msgs[bnode] = msg_val
                    except KeyError:
                        msgs[bnode] = msg_val

                # for each child in this level, send the messages to its parent (root)
                for snode in stree.keys():
                    # send messages to corresponding factor node
                    prod_msg = np.copy(prob[snode])
                    for mval in msgs[snode]:
                        prod_msg *= mval  # product over all msgs

                    # dot with potentials
                    if rnode == 1:
                        mval = pair_pot[1].dot(normalize_msg(prod_msg))
                    else:
                        mval = pair_pot[2].dot(normalize_msg(prod_msg))

                    try:
                        msgs[rnode].append(mval)
                    except KeyError:
                        msgs[rnode] = [mval]

            elif isinstance(stree, list):

                # For each leaf node, send msg to corresponding factor node
                for leaf in stree:
                    msgs[leaf] = np.copy(prob[leaf])

                for tup in fgraph[rnode]:
                    if tup[0] == 1:
                        val = pair_pot[1].dot(msgs[tup[1]])
                    else:
                        val = pair_pot[2].dot(msgs[tup[1]])

                    try:
                        msgs[tup[0]].append(val)
                    except KeyError:
                        msgs[tup[0]] = [val]

        return msgs

    def root_to_leaf(self, tree, fgraph, prob, pair_pot, beliefs):
        """ Send messages from root to leaf, using the beliefs that were obtained at root """

        # For each child node, send a msg from parent through the factor b/w them
        # Step 1: Product of all the msgs the parent got from other factors except
        # the one to which it is sending through now.
        # Step 2: Dot product the (factor matrix).T with the corresponding received msg and
        # forward it to child node.

        msgs = {}  # messages

        # print("tree:", tree)
        # print('fg:', fgraph)
        # print("probs:", prob)

        for rnode, stree in tree.items():
            print('n:', rnode, stree)
            if isinstance(stree, dict):
                # msgs from root to child nodes through factors
                for cnode in stree.keys():

                    prod_msg = np.copy(prob[rnode])

                    # product of messages that came to root except the one its sending to
                    for i, other_belief in enumerate(beliefs[rnode]):
                        if cnode == fgraph[rnode][i][1]:
                            # print('ignore this msg', cnode, i, fgraph[rnode][i])
                            continue
                        else:
                            prod_msg *= other_belief

                    # dot with factors.T
                    if rnode == 1:
                        msgs[cnode] = pair_pot[1].T.dot(normalize_msg(prod_msg))
                    else:
                        msgs[cnode] = pair_pot[2].T.dot(normalize_msg(prod_msg))

                pprint(msgs)
                top_msgs = self.root_to_leaf(stree, fgraph, prob, pair_pot, msgs)
                # print('top msgs:', top_msgs)
                # print('msgs:', msgs)
                for knode, msg_val in top_msgs.items():
                    msgs[knode] = msg_val

            elif isinstance(stree, list):
                for cnode in stree:
                    prod_msg = normalize_msg(prob[rnode] * beliefs[rnode])
                    # print('prod msg:', prod_msg)
                    if rnode == 1:
                        msgs[cnode] = pair_pot[1].T.dot(prod_msg)
                        print("Should not come here.")
                    else:
                        msgs[cnode] = pair_pot[2].T.dot(prod_msg)

        return msgs


    def send_message(self, f1, f2):
        """ Send message from factor 1 to 2 """

        pass

    def calculate_marginal(self, y):
        """ Calculate marginal probability for node y """

        pass

    def calculate_belief(self, y):
        """ Calculate belief for node y """

        pass


def main():
    """ main method """

    bprop = BeliefPropagation()
    tree, prob, pair_pot = bprop.get_tree()
    fgraph = bprop.construct_factor_graph(tree)

    msgs_r = bprop.leaf_to_root(tree, fgraph, prob, pair_pot)

    # print('MAIN: msgs:', msgs_r)

    belief_root = np.copy(prob[1])
    for mval in msgs_r[1]:
        belief_root *= mval

    # print(belief_root)
    # print("P(root)", belief_root / belief_root.sum())
    # print("--------------")

    msgs_l = bprop.root_to_leaf(tree, fgraph, prob, pair_pot, msgs_r)

    # print('MAIN: msgs_l', msgs_l)

    # for i in range(5, 10):
    #    print(i, normalize_msg(msgs_l[i] * prob[i]))


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description=__doc__)
    ARGS = PARSER.parse_args()
    main()

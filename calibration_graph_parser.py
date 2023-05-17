# This code is part of Tergite
#
# (C) Johan Blomberg, Gustav Grännsjö 2020
# (C) Copyright Miroslav Dobsicek 2020, 2021
# (C) Copyright David Wahlstedt 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import json

import networkx as nx
import redis

import settings
import utils.redis
from calibration.calibration_common import CALIBRATION_SUPERVISOR_PREFIX

# Set up redis connection
red = redis.Redis(decode_responses=True)

# For calibration graph Redis entries
CALIBRATION_GRAPH_PREFIX = f"{CALIBRATION_SUPERVISOR_PREFIX}:graph"

# Networkx representation of measurement graph
graph = nx.DiGraph()

# Topological node order of graph
topo_order = []


def parse_json(filename):
    with open(filename) as file:
        return json.load(file)


def create_graph_structure(nodes):
    """
    Creates a NetworkX graph from the JSON file(s). Checks that all graphs
    are free from cycles, and that all dependencies point to existing nodes.
    """
    global graph
    # List to keep track of node names
    names = []

    # Set up nodes
    for node in nodes:
        graph.add_node(node)
        names.append(node)

    # Now that all nodes exist, set up edges
    for node in nodes:
        deps = nodes[node]["dependencies"]

        for dep in deps:
            # All nodes this node depends on must be defined
            if dep not in names:
                raise ValueError(
                    f'Node "{node}" depends on node "{dep}", '
                    f'but the "{dep}" node has not been defined!'
                )

            # If the dependency node does exist, add a directed edge from the node to its dependency
            graph.add_edge(node, dep)

    # Graph is set up, ensure that it is acyclic
    try:
        cycle = nx.algorithms.find_cycle(graph)
        raise nx.exception.HasACycle(f"The graph contains a cycle! ({cycle})")
    except nx.exception.NetworkXNoCycle:
        # No cycles, good!
        pass

    # Remove redundant dependencies
    removed = 0
    print("Removing redundant dependencies...")
    for node in nodes:
        deps = nodes[node]["dependencies"]
        for dep in deps:
            # print(f"Paths for {node} to {dep}:")
            paths = nx.all_simple_paths(graph, node, dep, 8)
            for path in paths:
                # print(f"{path}")
                if len(path) > 2:
                    graph.remove_edge(node, dep)
                    removed += 1
                    # print(f"Removing edge {node}-{dep}")
                    break
    print(f"Done! Removed {removed} dependencies.")

    # Generate subgraph with the nodes reachable from the specified
    # goal nodes, given in settings.py:
    goals = settings.CALIBRATION_GOALS
    if goals:
        print(f"{goals=}")
        goal_nodes = set()
        for goal in goals:
            goal_nodes = goal_nodes | nx.descendants(graph, goal)
            goal_graph = graph.subgraph(goal_nodes | set(goals))

        # Henceforth, the graph to consider is what is reachable from
        # the goal nodes:
        graph = goal_graph

    # Topologically sort the graph to get a linear order we can go through it in
    global topo_order
    print("Starting topo sort...")
    topo_order = list(reversed(list(nx.topological_sort(graph))))
    print(f"{topo_order=}")

    print("Finished!")


def build_redis_nodes(nodes: dict):
    # Remove Redis entries we might have created previously
    utils.redis.del_keys(red, regex=f"^{CALIBRATION_GRAPH_PREFIX}")

    # Store topological node order

    # The global 'graph' variable now contains those nodes we selected
    # in create_graph_structure, by providing CALIBRATION_GOALS, and
    # these are going to be represented in Redis. If CALIBRATION_GOALS
    # was not set, the whole JSON file is used.

    # topo_order was generated from graph's contents, so it already
    # consists of the selected nodes
    for node in topo_order:
        red.rpush(f"{CALIBRATION_GRAPH_PREFIX}:topo_order", node)

    # Set up measurement nodes
    for node in graph.nodes():
        contents = nodes[node]
        # Main node info
        red.hset(
            f"{CALIBRATION_GRAPH_PREFIX}:measurement:{node}",
            "calibration_fn",
            contents["calibration_fn"],
        )
        red.hset(
            f"{CALIBRATION_GRAPH_PREFIX}:measurement:{node}",
            "check_fn",
            contents["check_fn"],
        )

        # Dependency info
        for dep_edge in graph.edges(node):
            red.rpush(f"{CALIBRATION_GRAPH_PREFIX}:dependencies:{node}", dep_edge[1])

        # Param data
        for goal_parameter in contents["goal_parameters"]:
            parameter_name = goal_parameter["name"]
            component = goal_parameter.get("component") or ""
            # Add the name of the goal parameter, i.e., the name of
            # the result quantity that is measured by the calibration
            # specified by the node
            red.rpush(
                f"{CALIBRATION_GRAPH_PREFIX}:goal_parameters:{node}", parameter_name
            )
            # Add the component type that the goal parameter concerns
            red.hset(
                f"{CALIBRATION_GRAPH_PREFIX}:goal_parameters:{node}:{parameter_name}",
                "component",
                component,
            )


if __name__ == "__main__":
    CALIBRATION_GRAPH = settings.CALIBRATION_GRAPH

    nodes = parse_json(CALIBRATION_GRAPH)

    create_graph_structure(nodes)

    build_redis_nodes(nodes)

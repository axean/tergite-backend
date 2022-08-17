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

import redis
import networkx as nx

# Set up redis connection
red = redis.Redis(decode_responses=True)

# Measurement parameter names
param_names = set()

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

    # Topologically sort the graph to get a linear order we can go through it in
    global topo_order
    print("Starting topo sort...")
    topo_order = reversed(list(nx.topological_sort(graph)))
    print("Finished!")


def build_redis_nodes(nodes):
    # Flush any old data
    # TODO: Remove only fields we create
    red.flushdb()

    # Store topological node order
    for node in topo_order:
        red.rpush("topo_order", node)

    # Set up measurement nodes
    for node in nodes:
        contents = nodes[node]
        # Main node info
        red.hset(f"measurement:{node}", "calibration_fn", contents["calibration_fn"])
        red.hset(f"measurement:{node}", "check_fn", contents["check_fn"])
        if "fidelity_measurement" in contents:
            red.hset(
                f"measurement:{node}",
                "fidelity_measurement",
                str(contents["fidelity_measurement"]),
            )
        else:
            red.hset(f"measurement:{node}", "fidelity_measurement", "False")

        # Dependency info
        for dep_edge in graph.edges(node):
            red.rpush(f"m_deps:{node}", dep_edge[1])

        # Param data
        for param in contents["params"]:
            p_name = param["name"]
            p_unit = param["unit"]
            p_thresh_upper = param["threshold_upper"]
            p_thresh_lower = param["threshold_lower"]
            p_timeout = param["timeout"]
            red.rpush(f"m_params:{node}", p_name)
            # Add measurement-specific parameter attributes
            red.hset(f"m_params:{node}:{p_name}", "unit", p_unit)
            red.hset(f"m_params:{node}:{p_name}", "threshold_upper", p_thresh_upper)
            red.hset(f"m_params:{node}:{p_name}", "threshold_lower", p_thresh_lower)
            red.hset(f"m_params:{node}:{p_name}", "timeout", p_timeout)
            param_names.add(p_name)


if __name__ == "__main__":
    DEFAULT_CALIBRATION_DAG = "calibration_graphs/default.json"
    BIG_CALIBRATION_DAG = "calibration_graphs/big_graph.json"

    nodes = parse_json(DEFAULT_CALIBRATION_DAG)

    create_graph_structure(nodes)

    build_redis_nodes(nodes)

# Created by Johan Blomberg and Gustav Grännsjö, 2020

import redis
import json
import networkx as nx

# Set up redis
red = redis.Redis(host="localhost", port=6379, decode_responses=True)

# measurement parameter names
param_names = set()

# Networkx representation of measurement graph
graph = nx.DiGraph()

# Topological node order of graph
topo_order = []


def parse_json(filename):
    with open(filename) as file:
        return json.load(file)


def create_graph_structure(definition):
    """
    Creates a NetworkX graph from the JSON file(s). Checks that all graphs
    are free from cycles, and that all dependencies point to existing nodes.
    """
    # List to keep track of node names
    names = []

    # Set up nodes
    for node in definition:
        graph.add_node(node)
        names.append(node)

    # Now that all nodes exist, set up edges
    for node in definition:
        deps = definition[node]["dependencies"]
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
    for node in definition:
        deps = definition[node]["dependencies"]
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


def build_redis_nodes(definition):
    # Flush any old data
    # TODO: Remove only fields we create
    red.flushdb()

    # Store topological node order
    for node in topo_order:
        red.rpush("topo_order", node)

    # Set up measurement nodes
    for node in definition:
        contents = definition[node]
        # Main node info
        red.hset(f"measurement:{node}", "cal_f", contents["cal_f"])
        red.hset(f"measurement:{node}", "check_f", contents["check_f"])
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
    nodes = parse_json("calibration_jsons/nodes.json")
    # nodes = parse_json('calibration_jsons/bigjson.json')

    create_graph_structure(nodes)

    build_redis_nodes(nodes)

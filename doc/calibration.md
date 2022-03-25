The automated calibration system has two primary files,
`calibration_graph_parser.py` and `calibration_supervisor.py`.
The JSON parser requires you to have a JSON file representing
all the measurements that are required to maintain your system
and how they relate to each other. Please note that the dependency 
structure has to be a DAG. See calibration_graphs/default.json
for an example of the syntax for this file.

The JSON parser will read the specified file and build your
measurement DAG within Redis.

Starting the calibration supervisor will regularly try to maintain
the system defined in Redis, by running a function `maintain()` on
each measurement node in a topological order. For an overview of
what `maintain()` does and the general algorithm, see
https://arxiv.org/abs/1803.03226

Each node defined in your JSON requires the names of two functions,
`calibration_fn` and `check_fn`. The `calibration_fn` function should
generate a job which will be sent to the `execution_worker` to
schedule a full measurement of the node parameters. The `check_fn`
should return a job which will schedule a more light-weight
measurement, which can be used to diagnose the status of the node. The
generated scenarios (sent from `execution_worker`) should include
three special tags. The first tag (index 0) is `job_id`. The tag in
slot 2 (index 1) should the script name, and slot 3 (index 2) should
be a boolean, set to `True`, indicating this job is requested from the
calibration supervisor. All such functions should be imported into the
calibration supervisor and stored in the `MEASUREMENT_JOBS`
dictionary.

In addition, for every measurement node, you will need to add a log
extraction method within the `postprocessing_worker` (either imported
or not), and store it in the `PROCESSING_METHODS` dictionary under the
key equal to the script name, identified with the corresponding
measurement node it will extract data from. This assumes that the
`calibration_fn` and `check_fn` jobs use the same structure for the
produced log files, so that the extraction method can be used for
both. If this is not the case, then they will have to use a different
name in tag slot 3 (index 2), and separate entries in
`PROCESSING_METHODS`. The extraction method should read the parameters
relevant to the measurement node and store them in Redis under the
`results:{node_name}:{parameter_name}` naming structure (not
fully implemented yet).

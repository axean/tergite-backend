The automated calibration system has two primary files,
calibration_json_parser.py and calibration_supervisor.
The json parser requires you to have a json file representing
all the measurements that are required to maintain your system
and how they relate to each other. Please note that the dependency 
structure has to be a DAG. See calibration_jsons/nodes.json
for an example of the syntax for this file.

The json parser will read the specified file and build your
measurement DAG within Redis.

Starting the calibration supervisor will regularly try to maintain
the system defined in Redis, by running a function maintain() on
each measurement node in a topological order. For an overview of
what maintain() does and the general algorithm, see
https://arxiv.org/abs/1803.03226

Each node defined in your json requires the names of two functions,
cal_f and check_f. The cal_f function should generate a Labber scenario
which will perform a full measurement of the node parameters. The
check_f should return a scenario which will perform a more light-weight
measurement which can be used to diagnose the status of the node. The
generated scenarios should include two special tags. The tag in slot 2
should be "calibration" and slot 3 should be the name of the measurement
node this function is connected to.
All such functions should be imported into the calibration supervisor
and stored in the MEASUREMENT_ROUTINES dictionary.

In addition, for every measurement node, you will need to add a log
extraction method within the postprocessing_worker (either imported or not),
and store it in the PROCESSING_METHODS dictionary under the key equal
to the measurement node it will extract data from. This assumes that
the cal_f and check_f scenarios use the same structure for their log files,
so that the extraction method can be used for both. If this is not the case,
then they will have to use a different name in tag slot 3 and separate entries
in PROCESSING_METHODS. The extraction method should read the parameters relevant
to the measurement node and store them in Redis under the "results:{parameter_name}"
naming structure.

#!/bin/bash

exit_with_error () {
    echo "Port configuration failed. Use BCC_PORT=<num> in the .env file."
    exit 1
}

# port handling
PORT_CONFIG=$(grep BCC_PORT= .env)               # eg: BCC_PORT=5000
PORT_NUMBER="${PORT_CONFIG#*=}"                  # extract the number
[[ -z "$PORT_NUMBER" ]]  &&  exit_with_error     # validation
[[ ! "$PORT_NUMBER" =~ ^[0-9]+$ ]]  &&  exit_with_error


# TODO:
# - clean up fixed paths from this script
# - they don't do any harm at the moment but they are confusing


# path (mostly for uvicorn; --app-dir option will come in a new release)
cd /home/dobsicek/repos/tergite-bcc

# clean start
rq empty pingu_job_preprocessing pingu_job_execution pingu_logfile_postprocessing
rm -fr /tmp/pingu

# worker processes
rq worker --path /home/dobsicek/repos/tergite-bcc pingu_job_preprocessing &
rq worker --path /home/dobsicek/repos/tergite-bcc pingu_job_execution &
rq worker --path /home/dobsicek/repos/tergite-bcc pingu_logfile_postprocessing &

# rest-api
uvicorn --host 0.0.0.0 --port "$PORT_NUMBER" rest_api:app --reload

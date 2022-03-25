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



# clean start
rq empty pingu_job_registration pingu_job_preprocessing pingu_job_execution pingu_logfile_postprocessing
rm -fr /tmp/pingu    # FIXME: Fixed path

# worker processes
rq worker pingu_job_registration &
rq worker pingu_job_preprocessing &
rq worker pingu_job_execution &
rq worker pingu_logfile_postprocessing &

# rest-api
uvicorn --host 0.0.0.0 --port "$PORT_NUMBER" rest_api:app --reload

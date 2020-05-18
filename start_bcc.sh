#!/bin/sh

# TODO:
# - convert this script to a supervisor deployment
# - paths and names should come from a configuration file

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
uvicorn --host 0.0.0.0 --port 5000 rest-api:app --reload

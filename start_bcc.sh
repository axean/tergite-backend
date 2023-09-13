#!/bin/bash

# Run this script as follows:
#
# - go to the root of the tergite-bcc rpository (this directory)
# - run ./start_bcc.sh --device [FILEPATH]
#
# where FILEPATH is a TOML file with device configuration. See
# backend_properties_config/device_test.toml for an example.
#
# For convenience, for instance do:
# ln -s FILEPATH ./device.toml
# where FILEPATH is the full path to the desired TOML file.

set -e # exit if any step fails

exit_with_error () {
    echo "Port configuration failed. Use BCC_PORT=<num> in the .env file."
    exit 1
}

# Port handling
PORT_CONFIG=$(grep BCC_PORT= .env)               # eg: BCC_PORT=5000
PORT_NUMBER="${PORT_CONFIG#*=}"                  # extract the number
[[ -z "$PORT_NUMBER" ]]  &&  exit_with_error     # validation
[[ ! "$PORT_NUMBER" =~ ^[0-9]+$ ]]  &&  exit_with_error



# Clean start
rq empty loke_job_registration loke_job_preprocessing loke_job_execution loke_logfile_postprocessing
rm -fr /tmp/loke    # FIXME: Fixed path


# Remove old Redis keys, by their prefixes
# - job_supervisor
# - calibration_supervisor
# - device properties
# - post-processing results
prefixes="job_supervisor calibration_supervisor postprocessing:results: device:"
for prefix in $prefixes
do
    echo deleting "\"$prefix*\"" from Redis
    for key in $(redis-cli --scan --pattern "$prefix*")
    do
        redis-cli del "$key" > /dev/null
    done
done

# Load configuration files, passing the command-line args to load_config
# for now load_config expects only one option: "--device DEVICE_FILE_PATH"
# other possible arguments in $@ will be passed back as "unrecognized_args",
# so they can be used if needed, in later parts of this script
unrecognized_args=$(python backend_properties_config/load_config.py "$@")

if test -n "$unrecognized_args"; then
   echo "Arguments unrecognized by load_config.py: $unrecognized_args, in case needed below"
fi

# Worker processes
rq worker loke_job_registration &
rq worker loke_job_preprocessing &
rq worker loke_job_execution &
rq worker loke_logfile_postprocessing &

# REST-API
uvicorn --host 0.0.0.0 --port "$PORT_NUMBER" rest_api:app --reload

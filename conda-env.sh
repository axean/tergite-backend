#!/bin/bash -e

echo "Loading conda rest-eve env"
eval "$(/mnt/extra_ssd/anaconda/anaconda3/bin/conda shell.bash hook)"
conda activate rest-eve

exec "$@"

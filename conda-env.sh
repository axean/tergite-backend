#!/bin/bash -e

echo "Loading conda tergite env"
eval "$(/usr/local/anaconda/anaconda3/condabin/conda shell.bash hook)"
conda activate tergite

exec "$@"

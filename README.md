# Installation

## Prerequisites
* Redis
* Labber, + export PYTHONPATH to Labber's Script module
* tqc-sf (Tergite-Quantify-Connector storage file) needs to be installed with pip (see section below)
* (optional) Use Anaconda environment. With Python 3.8.

## Installing Tergite-Quantify-Connector storage file
* clone this repository https://bitbucket.org/qtlteam/tergite-quantify-connector-storagefile/src/main/
* cd tergite-quantify-connector-storagefile
* pip install -e .

## Package installation
* cd tergite-bcc
* pip install -e .
* create .env file with BCC configuration. See dot-env-template.txt.

## Start
* ./start_bcc.sh

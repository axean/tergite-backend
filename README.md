# Installation

## Prerequisites
* Redis
* Labber, + export PYTHONPATH to Labber's Script module
* tqc-sf (Tergite-Quantify-Connector storage file) needs to be installed with pip (see section below)
* (optional) Use Anaconda environment. With Python 3.8.
* `qtanalysis` >= 0.3
    * Installation (provided a 'bcc' conda environment for BCC)
    ```
    conda activate bcc
    cd ~/repos
    git clone git@bitbucket.org:qtlteam/qtl-analysis.git
    cd qtl-analysis
    git checkout analysis_dev2
    pip install -e .
    ```
    Now this will be available in the BCC environement

## Installing Tergite-Quantify-Connector storage file
This installs tergite-quantify-connector-storagefile in edit mode to the current Python environment.

* `cd ~/repos`
* `git clone git@bitbucket.org:qtlteam/tergite-quantify-connector-storagefile.git`
* `cd tergite-quantify-connector-storagefile`
* `pip install -e .`

## Package installation
* cd tergite-bcc
* pip install -e .
* create .env file with BCC configuration. See dot-env-template.txt.

## Start
* ./start_bcc.sh

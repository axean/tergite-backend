# Installation

## Prerequisites
* Redis
* Labber, + export PYTHONPATH to Labber's Script module
* (optional) Use Anaconda environment. With Python 3.8.

## Package installation
* cd tergite-bcc
* pip install -e .
* create .env file with BCC configuration. See dot-env-template.txt.

## Start
* ./start_bcc.sh --device backend_properties_config/device_default.toml

## How to Run With Systemd

- Clone the repo

```shell
git clone git@bitbucket.org:qtlteam/tergite-bcc.git 
```

- Copy the `dot-env-template.txt` into the `.env` file and update the variables there in. Contact your teammates for
 the variables you are not sure of.

```shell
cd tergite-bcc
cp dot-env-template.txt .env
```

- In case you don't have labber installed, add dummy labber

```shell
mkdir Labber
echo "import typing" >> Labber/__init__.py
echo "LogFile = typing.Any" >> Labber/__init__.py
echo "Scenario = typing.Any" >> Labber/__init__.py
echo "ScriptTools = typing.Any" >> Labber/__init__.py
```

- Copy `bcc.service` to the systemd services folder

```shell
cp bcc.service /etc/systemd/system/bcc.service
```

- Get the path to your conda bin. Run the command below:

```shell
where conda
```

- Extract the conda bin path. Look for a path that is similar to `/home/{user}/anaconda3/bin/conda`, and remove the last part i.e. '/conda'

```shell
YOUR_CONDA_BIN_PATH=/home/johndoe/anaconda3/bin
```

- Extract also the path to this folder where `tergite-bcc` is.

```shell
YOUR_PATH_TO_BCC=$(pwd)
```

- Get also the current user

```shell
YOUR_USER=$(whoami)
```

- Replace the variables `YOUR_CONDA_BIN_PATH` and `YOUR_PATH_TO_BCC` with the right values in `/etc/systemd/system/bcc.service`

```shell
sudo sed -i "s:YOUR_USER:${YOUR_USER}:" /etc/systemd/system/bcc.service
sudo sed -i "s:YOUR_CONDA_BIN_PATH:${YOUR_CONDA_BIN_PATH}:" /etc/systemd/system/bcc.service
sudo sed -i "s:YOUR_PATH_TO_BCC:${YOUR_PATH_TO_BCC}:" /etc/systemd/system/bcc.service
```

- Start BCC service

```shell
sudo systemctl start bcc.service
```

- Check the BCC service status

```shell
sudo systemctl status bcc.service
```

- Enable BCC to start on startup incase the server is ever restarted.


```shell
sudo systemctl enable bcc.service
```

# Contributing to tergite-bcc

**This project is currently not accepting pull requests from the general public yet.**

**It is currently being developed by the core developers only.**

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

-   Reporting a bug
-   Discussing the current state of the code
-   Submitting a fix
-   Proposing new features
-   Becoming a maintainer

## Versioning

When versioning we follow the format `{year}.{month}.{patch_number}` e.g. `2023.12.0`.

## We Develop with Github

We use Github to host code, to track issues and feature requests, as well as accept pull requests.

But We Use [Github Flow](https://docs.github.com/en/get-started/quickstart/github-flow),
So All Code Changes Happen Through Pull Requests

Pull requests are the best way to propose changes to the codebase (we
use [Github Flow](https://docs.github.com/en/get-started/quickstart/github-flow)). We actively welcome your pull
requests:

1. Clone the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the Apache 2.0 Software Licenses

In short, when you submit code changes, your submissions are understood to be under the
same [Apache 2.0 License](./LICENSE.txt) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/tergite/tergite-bcc/issues)

We use Github issues to track bugs. Report a bug
by [opening a new issue](https://github.com/tergite/tergite-bcc/issues); it's that easy!

## Write bug reports with detail, background, and sample code

[This is an example](http://stackoverflow.com/q/12488905/180626).
Here's [another example from Craig Hockenberry](http://www.openradar.me/11905408).

**Great Bug Reports** tend to have:

-   A quick summary and/or background
-   Steps to reproduce
    -   Be specific!
    -   Give sample code if you can.
-   What you expected would happen
-   What actually happens
-   Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

People _love_ thorough bug reports. I'm not even kidding.

## License

By contributing, you agree that your contributions will be licensed under its Apache 2.0 License.

## Folder Structure

The root folder of this project has a number of files and folders that may not be necessary to consider if you are only
interested in the actual application. See the [`app/README.md`](./app/README.md) file for more information.

## How to Test

- Ensure you have a [redis server](https://redis.io/docs/install/install-redis/) installed on your local machine.
- Clone the repo and checkout the current branch

```shell
git clone git@bitbucket.org:qtlteam/tergite-bcc.git
cd tergite-bcc
git checkout enhancement/app-folder
```

- Create a conda environment with python 3.8

```shell
conda create -n bcc python=3.8
```

- Install requirements

```shell
conda activate bcc
pip install -r requirements.txt
```

- Start the redis server in another terminal

```shell
redis-server
```

- Lint with black

```shell
black --check app
```

- Run the tests by running the command below at the root of the project. 

```shell
pytest app
```

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

- Copy the hardware example config file `executor-config.example.yml` into the `executor-config.yml` file and update the variables there in. Contact your teammates for
 the variables you are not sure of.

```shell
cp quantum_executor-config.example.yml executor-config.yml
```

- Copy `bcc.service` to the systemd services folder

```shell
sudo cp bcc.service /etc/systemd/system/bcc.service
```

- Get the path to your conda bin:

```shell
YOUR_CONDA_BIN_PATH="${$(conda info --base)}/bin/conda"
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
sudo sed -i.bak "s:YOUR_USER:${YOUR_USER}:" /etc/systemd/system/bcc.service
sudo sed -i.bak "s:YOUR_CONDA_BIN_PATH:${YOUR_CONDA_BIN_PATH}:" /etc/systemd/system/bcc.service
sudo sed -i.bak "s:YOUR_PATH_TO_BCC:${YOUR_PATH_TO_BCC}:" /etc/systemd/system/bcc.service
sudo rm /etc/systemd/system/bcc.service.bak
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

## References

This document was adapted from [a gist by Brian A. Danielak](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62) which
was originally adapted from the open-source contribution guidelines
for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md)

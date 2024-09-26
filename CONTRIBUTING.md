# Contributing to tergite-backend

**This project is not accepting pull requests from the general public yet.**

**It is currently being developed by the core developers only.**

## Government Model

[Chalmers Next Labs AB (CNL)](https://chalmersnextlabs.se) manages and maintains this project on behalf of all contributors.

## Version Control

Tergite is developed on a separate version control system and mirrored publicly on GitHub.
If you are reading this on GitHub, then you are looking at a mirror. 

## Versioning

When versioning we follow the format `{year}.{month}.{patch_number}` e.g. `2023.12.0`.

## Contacting the Tergite Developers

Since the GitHub repositories are only mirrors, no GitHub pull requests or GitHub issue/bug reports 
are looked at. Please get in touch via email <quantum-nextlabs@chalmers.se> instead. 

Take note that the maintainers may not answer every email.

## But We Use [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow), So All Code Changes Happen Through Pull Requests

Pull requests are the best way to propose changes to the codebase (we
use [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)). We actively welcome your pull
requests:

1. Clone the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the Apache 2.0 software licenses

In short, when you submit code changes, your submissions are understood to be under the
same [Apache 2.0 License](./LICENSE.txt) that covers the project. Feel free to contact the maintainers if that's a concern.

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

## Contributor Licensing Agreement

Before you can submit any code, all contributors must sign a
contributor license agreement (CLA). By signing a CLA, you're attesting
that you are the author of the contribution, and that you're freely
contributing it under the terms of the Apache-2.0 license.

The [individual CLA](https://tergite.github.io/contributing/icla.pdf) document is available for review as a PDF.

Please note that if your contribution is part of your employment or 
your contribution is the property of your employer, 
you will also most likely need to sign a [corporate CLA](https://tergite.github.io/contributing/ccla.pdf).

All signed CLAs are emails to us at <quantum-nextlabs@chalmers.se>.

## How to Test

- Ensure you have a [redis server](https://redis.io/docs/install/install-redis/) installed on your local machine.
- Clone the repo

```shell
git clone git@github.com:tergite/tergite-backend.git
cd tergite-backend
```

- Create a conda environment with python 3.9

```shell
conda create -n bcc python=3.9
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
git clone git@github.com:tergite/tergite-backend.git
```

- Copy the `dot-env-template.txt` into the `.env` file and update the variables there in. Contact your teammates for
 the variables you are not sure of.

```shell
cd tergite-backend
cp dot-env-template.txt .env
```

- Copy the quantify example config file `quantify-config.example.yml` into the `quantify-config.yml` file and update the variables there in. Contact your teammates for
 the variables you are not sure of.

```shell
cp quantify-config.example.yml quantify-config.yml
```

- Copy `bcc.service` to the systemd services folder

```shell
sudo cp bcc.service /etc/systemd/system/bcc.service
```

- Get the path to your conda bin:

```shell
YOUR_CONDA_BIN_PATH="$(conda info --base)/bin"
```


- Extract also the path to this folder where `tergite-backend` is.

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

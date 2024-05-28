# Tergite Backend (formerly Tergite BCC)

![CI](https://github.com/tergite/tergite-backend/actions/workflows/ci.yml/badge.svg)

The Backend in the [Tergite software stack](https://tergite.github.io/) of the Swedish quantum computer.

**This project is developed by a core group of collaborators.**    
**Chalmers Next Labs AB (CNL) takes on the role of managing and maintaining this project.**

## Version Control

The tergite stack is developed on a separate version control system and mirrored on Github.
If you are reading this on GitHub, then you are looking at a mirror. 

## Dependencies

- [Python 3.9](https://www.python.org/)
- [Redis](https://redis.io/)

## Quick Start

- Ensure you have [conda](https://docs.anaconda.com/free/miniconda/index.html) installed. 
 (_You could simply have python +3.9 installed instead._)
- Ensure you have the [Redis](https://redis.io/) server running
- Clone the repo

```shell
git clone git@github.com:tergite/tergite-backend.git
```

- Create conda environment

```shell
conda create -n bcc -y python=3.9
conda activate bcc
```

- Install dependencies

```shell
cd tergite-backend
pip install -r requirements.txt
```

- Copy the `dot-env-template.txt` file to `.env` and 
  update the environment variables there appropriately.

```shell
cp dot-env-template.txt .env
```

- Copy the hardware example config file `executor-config.example.yml` into the `executor-config.yml` file and update the variables there in. Contact your teammates for
 the variables you are not sure of.

```shell
cp executor-config.example.yml executor-config.yml
```

_Note: If you want to just run a dummy cluster, you can copy the one in the test fixtures_

```shell
cp app/tests/fixtures/dummy-executor-config.yml executor-config.yml
```


- Run start script

```shell
./start_bcc.sh --device configs/device_default.toml
```

- Open your browser at [http://localhost:8000/docs](http://localhost:8000/docs) to see the interactive API docs

## Documentation

Find more documentation in the [docs folder](./docs)

## Contribution Guidelines

If you would like to contribute, please have a look at our
[contribution guidelines](./CONTRIBUTING.md)

## Authors

This project is a work of
[many contributors](https://github.com/tergite/tergite-backend/graphs/contributors).

Special credit goes to the authors of this project as seen in the [CREDITS](./CREDITS.md) file.

## ChangeLog

To view the changelog for each version, have a look at
the [CHANGELOG.md](./CHANGELOG.md) file.

## License

[Apache 2.0 License](./LICENSE.txt)

## Acknowledgements

This project was sponsored by:

-   [Knut and Alice Wallenburg Foundation](https://kaw.wallenberg.org/en) under the [Wallenberg Center for Quantum Technology (WAQCT)](https://www.chalmers.se/en/centres/wacqt/) project at [Chalmers University of Technology](https://www.chalmers.se)
-   [Nordic e-Infrastructure Collaboration (NeIC)](https://neic.no) and [NordForsk](https://www.nordforsk.org/sv) under the [NordIQuEst](https://neic.no/nordiquest/) project
-   [European Union's Horizon Europe](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en) under the [OpenSuperQ](https://cordis.europa.eu/project/id/820363) project
-   [European Union's Horizon Europe](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en) under the [OpenSuperQPlus](https://opensuperqplus.eu/) project

# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project follows versions of format `{year}.{month}.{patch_number}`.

## [Unreleased]

### Added

- Added redis connection environment variables
- Added GitLab CI configuration
- Added storing of Qobj header data in the logfiles of the quantum jobs

### Fixed

## [2024.09.1] - 2024-09-24

### Added

### Changed

### Fixed

- Fixed 'KeyError' when no units are not passed in the backend_config file
- Fixed "...bin/conda/activate: Not a directory" error when starting as systemd service
- Fixed silent error where calibrations are not sent to MSS on executor initialization
- Fixed "TypeError: Object of type datetime is not JSON serializable" when sending calibration data to MSS
- Fixed 'SyntaxWarning: 'is not' with a literal' when initializing backend

## [2024.09.0] - 2024-09-16

### Added
- The `QuantumExecutor` as abstract class to implement a backend
- `EXECUTOR_TYPE` keyword in the .env variables to select the backend
- Qiskit dynamics simulator backend with one qubit ("qiskit_pulse_1q")
- Added the initialization of the redis store with configuration picked from the `backend_config.toml` file
  when the execution worker starts
- Added an initial request to update the backend information in MSS 
  when the execution worker starts

### Changed
- BREAKING CHANGE: `EXECUTOR_DATA_DIRNAME` definition in the .env variables instead of `general.data_dir` in `executor-config.yml`
- BREAKING CHANGE: Removed the whole `general` section in the `executor-config.yml`
- BREAKING CHANGE: Renamed `executor-config.yml` to `quantify-config.yml`
- Removed the old config files that were used for setting up automatic calibration
- Removed the script that loaded automatic calibration configurations at the start
- Moved the `backend_config.toml` file from `/configs` folder
- Moved the `properties` service to the `libs` folder
- Moved the `storage`, `date_time`, `representation` and `logging` utils to the `properties` lib
- Removed the `scripts` folder
- Removed the `archive` folder

### Fixed

- Fixed the reporting to MSS of errors in jobs during post-processing

## [2024.04.0] - 2024-05-28

This is part of the tergite release v2024.04 which merges the tergite stack into three clear parts: front, back, client/sdks

### Added

- Added storage_file lib (formerly tergite-quantify-connector-storagefile)
- Added `quantum_executor` service (formerly tergite-quantify-connector)
- Added the `executor-config.yml` and its python-based validators
- Added a simulator for one transmon qubit `qiskit_pulse_1q`

### Changed

- Changed the way discriminators are loaded to load from the database
- BREAKING_CHANGE: Removed hard-coded discriminators
- BREAKING_CHANGE: Removed official support for Python 3.8; Official support is now >=3.9
- BREAKING_CHANGE: Removed Labber support
- Replaced tergite-quantify-connector-storagefile package with an internal storage_file lib
- Moved unused files to `archive` folder
- BREAKING_CHANGE: Removed calibration and two state discrimination source code
- BREAKING_CHANGE: Replaced tergite-quantify-connector-storagefile package with an internal storage_file lib
- BREAKING_CHANGE: Merged tergite-quantify-connector into tergite-backend and renamed its service to `quantum_executor`
- BREAKING_CHANGE: Changed configuration of hardware to use `executor-config.yml` file with proper validations on loading
- BREAKING_CHANGE: Removed support for `Pulsar`, or any other instrument drivers other than `Cluster`   
  The old implementation wrongfully assumed that all these drivers have the same signature i.e. `driver(name: str, identifier: str | None)`  
  yet `SpiRack(name: str, address: str, baud_rate: int = 9600, timeout: float = 1, is_dummy: bool = False,)`,   
  `Pulsar(name: str, identifier: Optional[str] = None, port: Optional[int] = None, debug: Optional[int] = None, dummy_type: Optional[PulsarType] = None,)`   
  `Cluster(name: str, identifier: Optional[str] = None, port: Optional[int] = None, debug: Optional[int] = None, dummy_type: Optional[PulsarType] = None)` are all different.  
- BREAKING_CHANGE: We got rid of quantify connector’s redundant reset() method.
- BREAKING_CHANGE: Changed backend name used when querying MSS for backend properties to be equal to `settings.DEFAULT_PREFIX`

### Fixed

- Fixed duplicate job uploads to respond with HTTP 409


## [2024.02.0] - 2024-03-19

This is part of the tergite release v2024.02 that introduces authentication, authorization and accounting

### Added

- Authenticated and authorized requests to/from clients including 
  [MSS](https://github.com/tergite/tergite-mss) and the general internet.
- Tracking of execution time of jobs

### Changed

### Fixed

### Contributors

- Martin Ahindura

## [2023.12.0] - 2024-03-11

This is part of the tergite release v2023.12.0 that is the last to support [Labber](https://www.keysight.com/us/en/products/software/application-sw/labber-software.html).
Labber is being deprecated.

### Added

- Initial release of the tergite-backend server
- Support for [Labber](https://www.keysight.com/us/en/products/software/application-sw/labber-software.html)
- Support for [quantify-core](https://quantify-os.org/docs/quantify-core)

### Changed

### Fixed

### Contributors

- Miroslav Dobsicek
- Abdullah-al Amin
- Stefan Hill
- Axel Andersson
- David Wahlstedt
- Fabian Forslund
- Nicklas Botö

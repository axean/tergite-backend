# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project follows versions of format `{year}.{month}.{patch_number}`.

## [Unreleased]

### Added

- Add a simpler JSON document based store in redis

### Changed

- Changed source of lda parameters to backend's redis store. Originally they were retrieved from MSS.
- Added proper HTTP status codes for failed REST API requests
  - InvalidJobIdInUploadedFileError results in a 400 HTTP response (originally was 200)
  - ItemNotFoundError results in a 404 HTTP response (originally was 200)
  - JobAlreadyCancelled results in a 406 HTTP response (originally was 500)

## [2025.03.2] - 2025-03-19

### Changed

- Remove stale fixtures 

### Fixed

- Fixed JSONDecodeError when application is run in systemd

## [2025.03.1] - 2025-03-18

### Changed

No change

## [2025.03.0] - 2025-03-18

### Changed

- Removed the storage-file internal lib
- Limited pyarrow to versions '18.0.0' and below for macOS
- Decouple data from algorithms in storage file
- Decouple native job run from executor instance
- Remove debug prints of qobj when running jobs
- Rename the `run_experiment` and `run` methods of the `QuantumEexcutor` class to `_run_native`, `run`
- Clean up qobj-to-quantify compilation
- Removed `Program`
- Removed the `dag` property of the `NativeExperiment` class
- Enchanced the `Channel` class to track all instructions attached to it
- Added the 'to_operation' method on the `BaseInstruction` of quantify, removing the long if-conditional `QauntifyExperiment.schedule_operation` method that was originally generating Operator's
- Replaced the looping of the DAG with the looping of the instructions on each channel
- Created the `ChannelRegistry` dict-like class to track the state of all channels attached to an experiment
- Deleted `rot_left` and `ceil4` utility functions and other unused utility functions
- Split `FreqInstruction` class to `SetFreqInstruction` and `ShiftFreqInstruction`
- Split `PhaseInstruction` class to `SetPhaseInstruction` and `ShiftPhaseInstruction`
- Added the `QBLOX_TIMEGRID_INTERVAL` constant
- Moved the `Instruction` class in the `quantum_executor/base` folder to `quantum_executor/quantify` folder, renaming it to `BaseInstruction`
- Removed the `channels` property from the `NativeExperiment` class
- Added the `channel_registry` property on the `QauntifyExperiment` class
- Moved the `Channel` definition from `utils` to the `quantum_executor/quantify` folder
- Removed `retworkx` from the requirements.txt
- Updated quantify-scheduler, quantify-core and qblox-instruments and pydantic versions
- BREAKING CHANGE: Split cluster configurations into `quantify-config.json` and `quantify-metadata.json`
- BREAKING CHANGE: Added a new configuration file `calibration.seed.toml` for seeding the database with calibration data
- Enhanced the `QuantifySchedule` conversion to use parametric schedules and new portclock convention


## [2024.12.2] - 2024-12-23

### Changed

- Changed the apt installation step in Dockerfile to remove `/var/lib/apt/lists/*` after completion
- Made removal of keys in `start_bcc.sh` optional

### Fixed

- Fixed docker error ModuleNotFoundError: No module named 'setuptools.command.build'

## [2024.12.1] - 2024-12-20

### Added

- Added Dockerfile.
- Added instructions how to run with docker.
- Added instructions how to run the qiskit_pulse_2q simulator in the configuration docs.

### Changed

- Changed `start_bcc.sh` script to use redis connection obtained from the environment.
- Changed `start_bcc.sh` script to update the exported environment variables after reading from the `ENV_FILE`.
- Updated Github action to deploy built multiplatform image to docker hub as something like `tergite/tergite-backend`
- Removed some redundant libraries in the `requirements.txt` file.
- Removed some outdated docs.

## [2024.12.0] - 2024-12-13

### Added

- Added redis connection environment variables
- Added GitLab CI configuration
- Added storing of Qobj header data in the logfiles of the quantum jobs
- Qiskit dynamics simulator backend with two-qubit CZ gate ("qiskit_pulse_2q")
- Added CouplerProps to Backend Configurations
- Added the `coupling_dict` to the `backend_config.toml`
- Added `Dockerfile` and `.dockerignore`

### Fixed

- Fixed httpx version to 0.27.2 as 0.28.0 removes many deprecations that we were still dependent on in FastAPI testClient

### Changed

- Removed the `coupling_map` from the `backend_config.toml` as it is generated on-the-fly from the `coupling_dict`.

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

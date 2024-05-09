# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project follows versions of format `{year}.{month}.{patch_number}`.

## [Unreleased]

This is part of the tergite release v2024.03 that updates the full pipeline for state discrimination hardware calibration

### Added

### Changed

- Changed the way discriminators are loaded to load from the database
- Removed hard-coded discriminators
- Upgrade to Python 3.9
- Removed Labber in job processing and calibration
- Replaced tergite-quantify-connector-storagefile package with an internal storage_file lib

### Fixed

### TODO

- [ ] Add quantify connector's main branch and push it both to upstream and downstream
- [ ] Merge quantify connector's simulator branch and push it to upstream
- [ ] Run simulator version on CTH side
- [ ] Run dummy cluster on local or any other place
- [ ] Test multiple BCC with MSS

### Contributors

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

- Initial release of the tergite-bcc server
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

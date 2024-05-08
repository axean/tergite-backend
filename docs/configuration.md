# Configuration

This is documentation on how to configure BCC


## QBLOX Instrument's Settings
Specification:

- The QBLOX instruments configurations should look like single units in the Quantify hardware configuration file.
- Generic QCoDeS instruments, such as local oscillators, should be a flat JSON blob where every key is a QCoDeS command for the device and the value corresponds to the set value. QCoDeS drivers can be found at qcodes.github.io. For example, here is the driver for SGS100A: https://qcodes.github.io/Qcodes/_modules/qcodes/instrument_drivers/rohde_schwarz/SGS100A.html.
- Every config file should additionally have keys "instrument_type" "instrument_address" and "instrument_driver". These are not QCoDeS commands, they are used by tqc to configure the instrument orchestration. QBLOX devices additionally need a key called "instrument_component" pointing to the IC component of the device.
- Important: Cluster module naming in hardware config: changing module names to <cluster_name>_module<slot_id> required! See: https://gitlab.com/quantify-os/quantify-scheduler/-/wikis/Qblox-ICCs:-Interface-changes-in-using-qblox-instruments-v0.6.0
- Important: When a new Cluster is instantiated, its component modules are automatically added using the naming convention `"<cluster_name>_module<slot>"`. See https://gitlab.com/quantify-os/quantify-scheduler/-/blob/main/quantify_scheduler/instrument_coordinator/components/qblox.py#L1027

See .json files for example.

## TODO

- [x] Choose between yaml, simplified JSON and toml for configuration: Maintain the same JSON config; but change the env file to use a simpler check for dummy cluster
- [x] Simplify hardware configuration using YAML. See [quantify docs](https://quantify-os.org/docs/quantify-scheduler/dev/reference/qblox/How%20to%20use.html#sec-qblox-how-to-configure).
- [ ] Rename some of the configurations to lower case values in quantify settings
- [ ] Simplify some functions e.g. configuration loading
- [ ] Write some tests for quantify
- [ ] Get rid of some global variables
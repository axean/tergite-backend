## Purpose
To define a data format that defines the configuration and properties of a 
backend system, as well as easy modification, resdis storage and serialized 
transmission of these values, so a webgui can be used for data visulaization.

## Working procedure 
The "bacend_properties_updater" module takes "backend_config.json" as a templet
config structure for a backend system. It updates the properties/keys values and
stors it in redis with the key "current_snapshot". 
The value update process takes place in the "update_NDUV" or "init_NDUV" methods 
in the "backend_properties_updater_utils.py", upto now it uses random values.

The "current_snapshot" is returned from "rest_api.py" when @get "webgui" or
"webgui/config" is requested.

## Test Usage
main.py will generate random data once evry 'sleep' time defiend in the main()
/test/test_backend_data_population.py will generate random data on each run. 
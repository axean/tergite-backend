## Adding a QuantumExecutor

The executor is the core component to run the quantum circuit on a backend.
It manages the access to the instruments or sets up the simulator in case the backend is a simulator.
Right now, we have implemented the following backends:
- "hardware": This is interfacing to qblox hardware via quantify and the recommended option, because the WACQT quantum processor is running on that platform. It can be run in the dummy mode, check out the [configuration manual](./configuration.md) on how to do it.
- "qutip": Interfaces to a [simulator from the AQP group on Chalmers](https://github.com/aqp-mc2-chalmers/chalmers-qubit), which is based on qutip. This is currently in experimental phase.
- "scqt": Interfaces to the OrangeQS simulator. This is not the preferred option, because it is proprietary and you would need a license.

If you want to implement a new executor, follow these steps below.

### 1. Create a class for your executor
- The base class for any quantum is `QuantumExecutor`.
- Inherit from that class to create your own executor.

### 2. Implement a transpiler
- Incoming jobs will be in the OpenPulse format.
- Implement a logic to translate the OpenPulse circuit to the format which your backend takes.
- This potentially involves interfacing the classes:
  - `BaseExperiment`: To implement a logic on how to iterate over the OpenPulse instructions.
  - `BaseProgram`: To translate the individual instructions.
- Other classes in the `quantum_executor.scheduler` module might have to be adapted.

### 3. Next steps
- If your backend needs some basic tune-up or training for a discriminator, you can add these actions in the `quantum_executor.scripts` module.
- Add documentation for your executor.
  - You probably have to update the `dot_env_template.txt` and the `executor-config.example.yml`.
- Add the executor to the factory such that it can be loaded from the execution worker.
- Check whether all tests are running.

For more intuition on how the executors are integrated into the stack, check out the existing ones in the source code.

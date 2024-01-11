# BCC (Backend Control Computer)

This is the application that runs on a classical computer that controls a quantum computer

## Folder Structure

If you just want to run BCC, just ignore all other folders on the root project except `app/`.
If BCC is to ever be extracted from this project, only the following will be necessary:

- app/
- settings.py
- dot-env-template.txt
- requirements.txt
- LICENSE.txt
- setup.py (maybe)
- start_bcc.sh
- backend_properties_config/ (probably)

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

- Copy the dot-env-template.txt to a .env file and make sure all variables there are updated.
- Run the tests by running the command below at the root of the project. 
- [Some 2 tests may fail because the tergite-quantify-connector-storage repo has not been update on its main branch yet bcc 
  has certain changes on its main branch that expect a new version of that dependency]

```shell
pytest app
```

## License

It is licensed under the [Apache 2.0](../LICENSE.txt) license.
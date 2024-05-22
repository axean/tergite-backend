from os import environ

from app.tests.utils.fixtures import get_fixture_path

TEST_DEFAULT_PREFIX = "test"
TEST_STORAGE_ROOT = "/tmp/jobs"

TEST_LOGFILE_DOWNLOAD_POOL_DIRNAME = "logfile_download_pool"
TEST_LOGFILE_UPLOAD_POOL_DIRNAME = "logfile_upload_pool"
TEST_JOB_UPLOAD_POOL_DIRNAME = "job_upload_pool"
TEST_JOB_EXECUTION_POOL_DIRNAME = "job_execution_pool"
TEST_JOB_PRE_PROC_POOL_DIRNAME = "job_pre_proc_pool"
TEST_STORAGE_PREFIX_DIRNAME = TEST_DEFAULT_PREFIX
TEST_JOB_SUPERVISOR_LOG = "job_supervisor.log"

TEST_MSS_MACHINE_ROOT_URL = "http://localhost:8002"
TEST_BCC_MACHINE_ROOT_URL = "http://localhost:8000"
TEST_BCC_PORT = 8000

TEST_MSS_APP_TOKEN = "some-mss-app-token-for-testing"

TEST_EXECUTOR_CONFIG_FILE = get_fixture_path("dummy-executor-config.yml")


def setup_test_env():
    """Sets up the test environment.

    It should be run before any imports
    """
    environ["APP_SETTINGS"] = "test"
    environ["DEFAULT_PREFIX"] = TEST_DEFAULT_PREFIX
    environ["STORAGE_ROOT"] = TEST_STORAGE_ROOT

    environ["LOGFILE_DOWNLOAD_POOL_DIRNAME"] = TEST_LOGFILE_DOWNLOAD_POOL_DIRNAME
    environ["LOGFILE_UPLOAD_POOL_DIRNAME"] = TEST_LOGFILE_UPLOAD_POOL_DIRNAME
    environ["JOB_UPLOAD_POOL_DIRNAME"] = TEST_JOB_UPLOAD_POOL_DIRNAME
    environ["JOB_EXECUTION_POOL_DIRNAME"] = TEST_JOB_EXECUTION_POOL_DIRNAME
    environ["JOB_PRE_PROC_POOL_DIRNAME"] = TEST_JOB_PRE_PROC_POOL_DIRNAME
    environ["STORAGE_PREFIX_DIRNAME"] = TEST_STORAGE_PREFIX_DIRNAME
    environ["JOB_SUPERVISOR_LOG"] = TEST_JOB_SUPERVISOR_LOG

    environ["MSS_MACHINE_ROOT_URL"] = TEST_MSS_MACHINE_ROOT_URL
    environ["BCC_MACHINE_ROOT_URL"] = TEST_BCC_MACHINE_ROOT_URL
    environ["BCC_PORT"] = f"{TEST_BCC_PORT}"

    environ["MSS_APP_TOKEN"] = TEST_MSS_APP_TOKEN
    environ["EXECUTOR_CONFIG_FILE"] = TEST_EXECUTOR_CONFIG_FILE
    environ["IS_AUTH_ENABLED"] = "True"

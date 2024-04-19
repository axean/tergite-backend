from datetime import datetime, timedelta, timezone

from NDUV import NDUV

from archive.backend_properties_updater.backend_properties_updater_utils import (
    TYPE1,
    update_NDUV,
)


# generate_value only returns values between the given lower
# and upper limit.
def test_new_value_in_range():
    lower_limit = 0.9
    upper_limit = 1.1
    delta = 10000
    update_rate = 1
    initial_value = 1
    initial_duv = NDUV(
        "property",
        datetime.now(timezone.utc) - timedelta(days=1),
        "Hz",
        initial_value,
        [TYPE1],
    )
    updated_duv = update_NDUV(initial_duv, update_rate, lower_limit, upper_limit, delta)
    assert lower_limit <= updated_duv.value <= upper_limit


# generate_value does not return a value with a difference larger
# than the specified delta from the given initial value.
def test_new_value_difference():
    lower_limit = 0
    upper_limit = 10000
    delta = 0.001
    update_rate = 1
    initial_value = 5000
    duv = NDUV(
        "property",
        datetime.now(timezone.utc) - timedelta(days=1),
        "Hz",
        initial_value,
        [TYPE1],
    )
    updated_DUV = update_NDUV(duv, update_rate, lower_limit, upper_limit, delta)
    assert abs(updated_DUV.value - initial_value) <= delta


# update_NDUV only generates a new value if the time since the last
# update is greater than update_rate.
def test_update_rate():
    lower_limit = 0
    upper_limit = 10
    delta = 1
    update_rate = 1
    initial_value = 5000
    last_update = datetime.now(timezone.utc) - timedelta(seconds=59)
    duv = NDUV("property", last_update, "Hz", initial_value, [TYPE1])
    not_updated_DUV = update_NDUV(duv, update_rate, lower_limit, upper_limit, delta)
    assert not_updated_DUV.date == last_update

    last_update = datetime.now(timezone.utc) - timedelta(minutes=1)
    duv = NDUV("property", last_update, "Hz", initial_value, [TYPE1])
    updated_DUV = update_NDUV(duv, update_rate, lower_limit, upper_limit, delta)
    assert updated_DUV.date != last_update

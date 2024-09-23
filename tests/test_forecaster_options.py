from forecasters.create import make_forecaster
from evaluation_utils.common_options import (
    parse_forecaster_options,
    get_forecaster_config,
)
from pathlib import Path

# This assumes the test file is in the tests/ directory
TEST_FORECASTER_PATH = Path(__file__).parent / "dummy_forecaster.py"


def test_make_forecaster_with_options():
    # Method 1: Provide options as separate key=value strings
    options_multiple = [
        "model=test-model",
        "test_option=A",
        "test_option=B",
        "test_option=C",
    ]
    parsed_options_multiple = parse_forecaster_options(options_multiple)
    config_multiple = get_forecaster_config(None, options_multiple)

    forecaster_multiple = make_forecaster(
        forecaster_class=None,
        custom_path=str(TEST_FORECASTER_PATH) + "::DummyForecaster",
        forecaster_config=config_multiple,
    )
    dumped_config_multiple = forecaster_multiple.dump_config()

    # Method 2: Provide options as a single list
    options_list = ["model=test-model", "test_option=[A, B, C]"]
    parsed_options_list = parse_forecaster_options(options_list)
    config_list = get_forecaster_config(None, options_list)

    forecaster_list = make_forecaster(
        forecaster_class=None,
        custom_path=str(TEST_FORECASTER_PATH) + "::DummyForecaster",
        forecaster_config=config_list,
    )
    dumped_config_list = forecaster_list.dump_config()

    print(f"{dumped_config_multiple=}")
    print(f"{dumped_config_list=}")

    # Assertions
    assert dumped_config_multiple == dumped_config_list
    assert dumped_config_multiple["model"] == "test-model"
    assert dumped_config_multiple["test_option"] == ["A", "B", "C"]

    # Check that both parsing methods result in the same config
    assert parsed_options_multiple == parsed_options_list
    assert config_multiple == config_list

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import uuid
from common.llm_utils import (
    query_api_chat_sync,
    query_api_chat,
    query_parse_last_response_into_format_sync,
    query_parse_last_response_into_format,
    query_api_chat_native,
    query_api_chat_sync_native,
)

from common.llm_utils import Example
from common.datatypes import ForecastingQuestion, Forecast
from forecasters import (
    BasicForecaster,
    CoT_Forecaster,
    BasicForecasterTextBeforeParsing,
    CoT_ForecasterTextBeforeParsing,
)

default_small_model = "gpt-4o-mini-2024-07-18"

mock_q = "Will Manhattan have a skyscraper a mile tall by 2030?"
mock_a = "0.03"
mock_response_list = ["0.03", "0.05", "0.02"]
mock_response = MagicMock(prob=0.09)
mock_cot_response = MagicMock(
    chain_of_thought="I think because of this and that, the probability is 0.09",
    prob=0.09,
)


@pytest.fixture
def basic_forecaster():
    examples = [mock_q]
    return BasicForecaster(preface="Test preface", model=default_small_model)


@pytest.fixture
def mock_forecasting_question():
    return ForecastingQuestion(
        id=uuid.uuid4(),
        title="Test Title",
        body="Test Body",
        question_type="binary",
        resolution_date=datetime(2024, 1, 1),
        data_source="synthetic",
        url="http://example.com",
        metadata={"topics": ["test"]},
        resolution=True,
    )


actual_fq = ForecastingQuestion(
    title="Will Manhattan have a skyscraper a mile tall by 2030?",
    body=(
        "Resolves YES if at any point before 2030, there is at least "
        "one building in the NYC Borough of Manhattan (based on current "
        "geographic boundaries) that is at least a mile tall."
    ),
    resolution_date="2030-01-01T00:00:00",
    question_type="binary",
    created_date=None,
    data_source="manifold",
    url="https://www.metaculus.com/questions/12345/",
    metadata={"foo": "bar"},
    resolution=None,
)


@patch("forecasters.basic_forecaster.answer_sync", return_value=mock_response)
def test_basic_forecaster_call(
    mock_answer_sync, basic_forecaster, mock_forecasting_question
):
    expected_prob = mock_response.prob
    forecast = basic_forecaster.call_full(mock_forecasting_question)
    assert forecast.prob == pytest.approx(
        expected_prob
    ), "The calculated probability does not match the expected value"
    mock_answer_sync.assert_called_once()


@pytest.mark.asyncio
@patch("forecasters.basic_forecaster.answer", return_value=mock_response)
async def test_basic_forecaster_call_async(
    mock_answer, basic_forecaster, mock_forecasting_question
):
    expected_prob = mock_response.prob
    forecast = await basic_forecaster.call_async_full(mock_forecasting_question)
    assert forecast.prob == pytest.approx(
        expected_prob
    ), "The calculated probability does not match the expected value"
    mock_answer.assert_called_once()


def test_basic_forecaster_actual_call(mock_forecasting_question):
    # Create BasicForecaster instance
    forecaster = BasicForecaster(model=default_small_model)

    config = forecaster.dump_config()
    print(f"\n{config=}")
    assert isinstance(config, dict)
    assert "model" in config
    assert "preface" in config
    assert "examples" in config

    # Call the forecaster with actual prompts
    forecast = forecaster.call_full(actual_fq)

    # Print the forecast for manual inspection
    print(f"\nForecast: {forecast}")

    # Assert that we got a result
    assert isinstance(
        forecast, Forecast
    ), "Expected a Forecast object, but got a different type"
    assert isinstance(
        forecast.prob, float
    ), "Expected a float probability, but got a different type"
    assert (
        0 <= forecast.prob <= 1
    ), f"Probability {forecast.prob} is not between 0 and 1"
    assert forecast.metadata is None, "Expected metadata to be None"


def test_basic_forecaster_actual_call_with_examples(mock_forecasting_question):
    # Prepare test data
    user_preface = (
        "You are an informed and well-calibrated forecaster. I need you to give me "
        "your best probability estimate for the following sentence or question resolving YES. "
        "Your answer should be a float between 0 and 1, with nothing else in your response."
    )
    examples = [Example(user=mock_q, assistant=mock_a)]
    forecaster = BasicForecaster(
        preface=user_preface, examples=examples, model=default_small_model
    )

    config = forecaster.dump_config()
    print(f"\n{config=}")
    assert isinstance(config, dict)
    assert "model" in config
    assert "preface" in config and config["preface"] == user_preface
    assert "examples" in config

    forecast = forecaster.call_full(mock_forecasting_question)

    # Print the forecast for manual inspection
    print(f"\nForecast: {forecast}")


def test_cot_forecaster_actual_call(mock_forecasting_question):
    # Prepare test data
    user_preface = (
        "What's the probability of this event occurring? Start with an initial estimate. You MUST mention the words 'Initial Estimate' in your answer\n"
        "Consider some counterarguments to your initial estimate. You MUST mention the word 'Counterarguments' in your answer\n"
        "Given these counterarguments, what's your final probability estimate?\n"
        "It MUST change from your initial estimate, at least a little bit. Even 0.001 of a percent change is enough.\n"
        "You MUST mention the word 'Final' in your answer\n"
        "Your chain of thought MUST end with a number between 0 and 1, inclusive."
    )

    # Call the forecaster with actual prompts
    forecaster = CoT_Forecaster(
        preface=user_preface, examples=None, model=default_small_model
    )

    config = forecaster.dump_config()
    print(f"\n{config=}")
    assert isinstance(config, dict)
    assert "model" in config
    assert "preface" in config
    assert "examples" in config

    forecast = forecaster.call_full(mock_forecasting_question)

    # Print the chain of thought for manual inspection
    print(f"\n{forecast.metadata['chain_of_thought']=}")
    print(f"{forecast.prob=}\n")

    # Assert that we got a result
    assert isinstance(
        forecast.prob, float
    ), "Expected a float probability, but got a different type"
    assert (
        0 <= forecast.prob <= 1
    ), f"Probability {forecast.prob} is not between 0 and 1"

    # Verify that the chain of thought was captured and shows a change in opinion
    assert (
        "initial estimate" in forecast.metadata["chain_of_thought"].lower()
    ), "Chain of thought doesn't mention an initial estimate"
    assert (
        "counterarguments" in forecast.metadata["chain_of_thought"].lower()
    ), "Chain of thought doesn't mention counterarguments"
    assert (
        "final" in forecast.metadata["chain_of_thought"].lower()
    ), "Chain of thought doesn't mention a final estimate"

    # assert that the chain of thought final answer changes
    # Extract the final probability from the chain of thought
    import re

    cot_implied_prob_str = re.findall(
        r"\d+\.\d+", forecast.metadata["chain_of_thought"]
    )[-1]
    try:
        cot_implied_prob = float(cot_implied_prob_str)
        print(f"{cot_implied_prob=}")
        assert (
            abs(cot_implied_prob - forecast.prob) < 1e-6
        ), f"Final probability in chain of thought ({cot_implied_prob}) doesn't match returned probability ({forecast.prob})"
    except ValueError:
        pytest.fail(
            f"Failed to extract a valid final probability from the chain of thought. Last word was: {cot_implied_prob_str}"
        )


def test_crowd_forecaster():
    from forecasters import CrowdForecaster, Forecaster

    # Create mock forecasters
    forecaster1 = MagicMock(spec=Forecaster)
    forecaster2 = MagicMock(spec=Forecaster)

    # Define mock responses
    mock_fq = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Test Crowd Forecast",
        body="Test Body",
        question_type="binary",
        resolution_date=datetime(2025, 1, 1),
        data_source="synthetic",
        url="http://example.com",
        metadata={"topics": ["crowd_test"]},
        resolution=None,
    )
    probs = [0.6, 0.8]
    weights = [1, 2]
    forecast1 = Forecast(prob=probs[0])
    forecast2 = Forecast(prob=probs[1])

    forecaster1.call.return_value = forecast1
    forecaster2.call.return_value = forecast2

    # Initialize CrowdForecaster
    crowd_forecaster = CrowdForecaster(
        forecasters=[forecaster1, forecaster2], method="mean", weights=weights
    )

    config = crowd_forecaster.dump_config()
    print(f"\n{config=}")
    assert (
        isinstance(config, dict)
        and "forecasters" in config
        and "method" in config
        and "weights" in config
    )

    # Call CrowdForecaster
    combined_forecast = crowd_forecaster.call(mock_fq)
    print(f"\n{combined_forecast.prob=:.3f}")

    # Assert combined probability
    assert (
        combined_forecast.prob
        == pytest.approx((probs[0] * weights[0] + probs[1] * weights[1]) / sum(weights))
    ), "Combined probability should be the weighted mean of the individual probabilities"
    assert combined_forecast.metadata["probs"] == [probs[0], probs[1]]

    forecaster1.call.assert_called_once_with(mock_fq)
    forecaster2.call.assert_called_once_with(mock_fq)


def test_crowd_forecaster_extremize():
    from forecasters import CrowdForecaster, Forecaster

    # Create mock forecasters
    forecaster1 = MagicMock(spec=Forecaster)
    forecaster2 = MagicMock(spec=Forecaster)
    forecaster3 = MagicMock(spec=Forecaster)

    # Define mock responses
    mock_fq = ForecastingQuestion(
        id=uuid.uuid4(),
        title="Test Crowd Forecast",
        body="Test Body",
        question_type="binary",
        resolution_date=datetime(2025, 1, 1),
        data_source="synthetic",
        url="http://example.com",
    )
    forecast1 = Forecast(prob=0.8)
    forecast2 = Forecast(prob=0.8)
    forecast3 = Forecast(prob=0.8)

    forecaster1.call.return_value = forecast1
    forecaster2.call.return_value = forecast2
    forecaster3.call.return_value = forecast3

    # Initialize CrowdForecaster
    crowd_forecaster = CrowdForecaster(
        forecasters=[forecaster1, forecaster2, forecaster3],
        method="mean",
        extremize_alpha=1.5,
    )

    combined_forecast = crowd_forecaster.call(mock_fq)
    print(f"\n{combined_forecast.prob=:.3f}")
    assert (
        combined_forecast.prob < 1
    ), "Combined extremizedprobability should be less than 1"
    assert (
        combined_forecast.prob > 0.8
    ), "Combined extremized probability should be greater than 0.8"

    assert combined_forecast.metadata["probs"] == [
        forecast1.prob,
        forecast2.prob,
        forecast3.prob,
    ]

    forecaster1.call.assert_called_once_with(mock_fq)
    forecaster2.call.assert_called_once_with(mock_fq)
    forecaster3.call.assert_called_once_with(mock_fq)


@pytest.fixture
def basic_forecaster_text_before_parsing():
    return BasicForecasterTextBeforeParsing(
        model="gpt-4o-2024-08-06", examples=None, parsing_model="gpt-4o-mini-2024-07-18"
    )


@pytest.fixture
def cot_forecaster_text_before_parsing():
    return CoT_ForecasterTextBeforeParsing(
        model="gpt-4o-2024-08-06", examples=None, parsing_model="gpt-4o-mini-2024-07-18"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forecaster_fixture",
    ["basic_forecaster_text_before_parsing", "cot_forecaster_text_before_parsing"],
)
@patch("common.llm_utils.query_api_chat_native", wraps=query_api_chat_native)
@patch(
    "common.llm_utils.query_parse_last_response_into_format",
    wraps=query_parse_last_response_into_format,
)
@patch("common.llm_utils.query_api_chat", wraps=query_api_chat)
async def test_forecaster_text_before_parsing_actual_call_async(
    mock_query_api_chat_native,
    mock_query_parse_last_response_into_format,
    mock_query_api_chat,
    forecaster_fixture,
    request,
):
    forecaster = request.getfixturevalue(forecaster_fixture)

    # Mocks wrap the actual LLM calls
    forecast = await forecaster.call_async(actual_fq)
    assert isinstance(forecast, Forecast)
    assert isinstance(forecast.prob, float)
    assert 0 <= forecast.prob <= 1
    if isinstance(forecaster, CoT_ForecasterTextBeforeParsing):
        assert "chain_of_thought" in forecast.metadata
    assert mock_query_api_chat_native.call_count == 1
    assert mock_query_parse_last_response_into_format.call_count == 1
    assert mock_query_api_chat.call_count == 1
    mock_query_api_chat_native.reset_mock()
    mock_query_parse_last_response_into_format.reset_mock()
    mock_query_api_chat.reset_mock()


@pytest.mark.parametrize(
    "forecaster_fixture",
    ["basic_forecaster_text_before_parsing", "cot_forecaster_text_before_parsing"],
)
@patch("common.llm_utils.query_api_chat_sync_native", wraps=query_api_chat_sync_native)
@patch(
    "common.llm_utils.query_parse_last_response_into_format_sync",
    wraps=query_parse_last_response_into_format_sync,
)
@patch("common.llm_utils.query_api_chat_sync", wraps=query_api_chat_sync)
def test_forecaster_text_before_parsing_actual_call_sync(
    mock_query_api_chat_sync_native,
    mock_query_parse_last_response_into_format_sync,
    mock_query_api_chat_sync,
    forecaster_fixture,
    request,
):
    forecaster = request.getfixturevalue(forecaster_fixture)

    # Mocks wrap the actual LLM calls
    forecast = forecaster.call(actual_fq)
    assert isinstance(forecast, Forecast)
    assert isinstance(forecast.prob, float)
    assert 0 <= forecast.prob <= 1
    if isinstance(forecaster, CoT_ForecasterTextBeforeParsing):
        assert "chain_of_thought" in forecast.metadata
    assert mock_query_api_chat_sync_native.call_count == 1
    assert mock_query_parse_last_response_into_format_sync.call_count == 1
    assert mock_query_api_chat_sync.call_count == 1
    mock_query_api_chat_sync_native.reset_mock()
    mock_query_parse_last_response_into_format_sync.reset_mock()
    mock_query_api_chat_sync.reset_mock()

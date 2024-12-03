import pytest
import re
import os
from dotenv import load_dotenv

from perplexity_resolver import (
    parse_xml_resolver_output,
    parse_resolver_output,
    resolve_question,
)
from common.datatypes import ResolverOutput

# Load environment variables
load_dotenv()


pytest.mark.expensive = pytest.mark.skipif(
    os.getenv("TEST_PERPLEXITY_RESOLVER", "False").lower() == "false",
    reason="Skipping expensive perplexity resolver tests",
)


def og_use_openrouter():
    """
    Returns None if OPENROUTER_API_KEY is not available
    """
    original_openrouter_api_key = os.getenv("OPENROUTER_API_KEY", None)
    if not original_openrouter_api_key:
        return None
    original_use_openrouter = os.getenv("USE_OPENROUTER", "False")
    os.environ["USE_OPENROUTER"] = "True"
    return original_use_openrouter


# All functions are now top-level


def test_parse_xml_resolver_output_success():
    original_use_openrouter = og_use_openrouter()
    if original_use_openrouter is None:
        pytest.skip("OPENROUTER_API_KEY not available")
    test_input = """
    <resolver_output>
      <chain_of_thought>This is a test thought.</chain_of_thought>
      <can_resolve_question>true</can_resolve_question>
      <answer>true</answer>
    </resolver_output>
    """
    result = parse_xml_resolver_output(test_input)
    assert isinstance(result, ResolverOutput)
    assert result.chain_of_thought == "This is a test thought."
    assert result.can_resolve_question is True
    assert result.answer is True

    os.environ["USE_OPENROUTER"] = original_use_openrouter


def test_parse_xml_resolver_output_failure():
    test_input = "Invalid XML"
    with pytest.raises(ValueError):
        parse_xml_resolver_output(test_input)


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_resolve_question_with_malformed_xml():
    original_use_openrouter = og_use_openrouter()
    if original_use_openrouter is None:
        pytest.skip("OPENROUTER_API_KEY not available")
    question_title = "Will the 2024 Summer Olympics in Paris have over 10,000 athletes participating?"
    full_string = """
something
<resolver_output>
  <chai_of_thought>
    1. IOC set quota of 10,500 athletes.
    2. Organizers adhering to quota.
    3. No significant changes reported.
    Likely close to, but not exceeding, 10,000 athletes.
  </chain_of_thought>
  <can_resolve_question>true</can_resolve_question>
  <answer>false</answer>
</resolver_output>
    """

    expected_output = ResolverOutput(
        chain_of_thought="1. IOC set quota of 10,500 athletes. 2. Organizers adhering to quota. 3. No significant changes reported. Likely close to, but not exceeding, 10,000 athletes.",
        can_resolve_question=True,
        answer=False,
    )

    result = await parse_resolver_output(full_string, question_title)
    assert isinstance(result, ResolverOutput)

    # Clean up and compare chain_of_thought
    def clean_string(s):
        return re.sub(r"\s+", "", s.lower())

    assert clean_string(result.chain_of_thought) == clean_string(
        expected_output.chain_of_thought
    )

    assert result.can_resolve_question is expected_output.can_resolve_question
    assert result.answer is expected_output.answer

    os.environ["USE_OPENROUTER"] = original_use_openrouter


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_resolve_true_question():
    original_use_openrouter = og_use_openrouter()
    if original_use_openrouter is None:
        pytest.skip("OPENROUTER_API_KEY not available")
    question_title = (
        "Will Ireland win 4 or more gold medals at the 2024 Paris Olympics?"
    )
    question_body = """
This question will resolve Yes if Ireland wins 4 or more gold medals at the 2024 Summer Olympics in Paris. It will resolve No if Ireland wins 3 or fewer gold medals. The final count will be based on the official medal tally at the conclusion of the Games.
    """

    result = await resolve_question(
        question_title,
        question_body,
        resolution_date="2024-08-06",
        created_date="2024-05-30",
    )
    assert isinstance(result, ResolverOutput)

    # weaker assertions, it's an online model
    assert (
        result.can_resolve_question is True
    ), "Our Perplexity pipeline did not find a resolution to the question. This could be a bug, but can sometimes happen through no fault of our own, because the model is online, the internals might be updated, and performance varies."
    assert (
        result.answer is True
    ), "Our Perplexity pipeline resolved the question incorrectly. This is bad."

    os.environ["USE_OPENROUTER"] = original_use_openrouter


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_resolve_false_question():
    original_use_openrouter = og_use_openrouter()
    if original_use_openrouter is None:
        pytest.skip("OPENROUTER_API_KEY not available")
    question_title = (
        "Will Ireland win 6 or more gold medals at the 2024 Paris Olympics?"
    )
    question_body = """
This question will resolve Yes if Ireland wins 6 or more gold medals at the 2024 Summer Olympics in Paris. It will resolve No if Ireland wins 5 or fewer gold medals. The final count will be based on the official medal tally at the conclusion of the Games.
    """

    result = await resolve_question(
        question_title, question_body, resolution_date="2024-08-06", created_date=""
    )
    assert isinstance(result, ResolverOutput)

    assert (
        result.can_resolve_question is True
    ), "Our Perplexity pipeline did not find a resolution to the question. This could be a bug, but can sometimes happen through no fault of our own, because the model is online, the internals might be updated, and performance varies."
    assert (
        result.answer is False
    ), "Our Perplexity pipeline resolved the question incorrectly. This is bad."

    os.environ["USE_OPENROUTER"] = original_use_openrouter

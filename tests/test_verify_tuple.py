import pytest
from datetime import datetime
from uuid import uuid4
from common.datatypes import ForecastingQuestion, VerificationResult
from static_checks.MiniInstantiator import Neg, And, Trivial, Or, Paraphrase, Conditional, Consequence


skip_test_trivial_verify = False
skip_neg_verify_pass = False
skip_neg_verify_fail = False
skip_and_verify_pass = False
skip_and_verify_fail = False
skip_or_verify_pass = False
skip_or_verify_pass = False
skip_paraphrase_verify_pass = False
skip_paraphrase_verify_fail = False
skip_conditional_verify_pass = False
skip_conditional_verify_fail = False
skip_consequence_quantity_verify_pass = False
skip_consequence_quantity_verify_fail = False
skip_consequence_time_verify_pass = False
skip_consequence_time_verify_fail = False
skip_consequence_misc_verify_pass = False
skip_consequence_misc_verify_fail = False

@pytest.mark.asyncio
@pytest.mark.skipif(skip_test_trivial_verify, reason="Skipping trivial verify test")
async def test_trivial_verify():
    trivial_instance = Trivial()
    
    sample_question = ForecastingQuestion(
        id=uuid4(),
        title="Will the stock market close higher tomorrow?",
        body="Resolves YES if the stock market closes higher than today. Resolves NO otherwise.",
        resolution_date=datetime(2024, 8, 10),
        question_type="binary"
    )
    
    base_sentences = Trivial.BaseSentenceFormat(P=sample_question)
    output = Trivial.OutputFormat(P=sample_question)
    
    result = await trivial_instance.verify(output, base_sentences)
    
    assert result.valid == True, "Verification should always pass for Trivial class"
    assert isinstance(result, VerificationResult), "Result should be an instance of VerificationResult"

    result_sync = trivial_instance.verify_sync(output, base_sentences)
    
    assert result_sync.valid == True, "Verification should always pass for Trivial class, sync version"
    assert isinstance(result_sync, VerificationResult), "Result should be an instance of VerificationResult, sync version"


@pytest.mark.asyncio
@pytest.mark.skipif(skip_neg_verify_pass, reason="Skipping negation verify test")
async def test_neg_verify_pass():
    neg_instance = Neg()
    
    sample_question = ForecastingQuestion(
        id=uuid4(),
        title="Will the price of Bitcoin be above $100,000 on 1st January 2025?",
        body="Resolves YES if the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025. Resolves NO otherwise.",
        resolution_date=datetime(2025, 1, 1),
        question_type="binary"
    )
    
    sample_output = Neg.OutputFormat(
        not_P=ForecastingQuestion(
            id=uuid4(),
            title="Will the price of Bitcoin be less than or equal to $100,000 on 1st January 2025?",
            body="Resolves YES if the spot price of Bitcoin against USD is less than or equal to 100,000 on 1st January 2025. Resolves NO otherwise.",
            resolution_date=datetime(2025, 1, 1),
            question_type="binary"
        )
    )
    
    base_sentences = Neg.BaseSentenceFormat(P=sample_question)
    
    result = await neg_instance.verify(sample_output, base_sentences)
    
    assert result.valid == True, f"Verification failed. Reasoning: {result.reasoning}"
    assert len(result.reasoning) > 0, "Verification reasoning should not be empty"

    result_sync = neg_instance.verify_sync(sample_output, base_sentences)

    assert result_sync.valid == True, f"Verification failed. Reasoning: {result.reasoning}, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning should not be empty, sync version"

@pytest.mark.asyncio
@pytest.mark.skipif(skip_neg_verify_fail, reason="Skipping negation verify test")
async def test_neg_verify_fail():
    neg_instance = Neg()
    
    sample_question = ForecastingQuestion(
        id=uuid4(),
        title="Will the price of Bitcoin be above $100,000 on 1st January 2025?",
        body="Resolves YES if the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025. Resolves NO otherwise.",
        resolution_date=datetime(2025, 1, 1),
        question_type="binary"
    )
    
    incorrect_output = Neg.OutputFormat(
        not_P=ForecastingQuestion(
            id=uuid4(),
            title="Will the price of Bitcoin be exactly $100,000 on 1st January 2025?",
            body="Resolves YES if the spot price of Bitcoin against USD is exactly 100,000 on 1st January 2025. Resolves NO otherwise.",
            resolution_date=datetime(2025, 1, 1),
            question_type="binary"
        )
    )
    print("not p title = ", incorrect_output.not_P.title) 
    base_sentences = Neg.BaseSentenceFormat(P=sample_question)
    
    result = await neg_instance.verify(incorrect_output, base_sentences)
    
    assert result.valid == False, "Verification should fail for incorrect negation"
    assert len(result.reasoning) > 0, "Verification reasoning for failure should not be empty"

    result_sync = neg_instance.verify_sync(incorrect_output, base_sentences)

    assert result_sync.valid == False, "Verification should fail for incorrect negation, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning for failure should not be empty, sync version"

@pytest.mark.asyncio
@pytest.mark.skipif(skip_and_verify_pass, reason="Skipping AND verify test")
async def test_and_verify_pass():
    and_instance = And()
    
    sample_question_1 = ForecastingQuestion(
        id=uuid4(),
        title="Will the price of Bitcoin be above $100,000 on 1st January 2025?",
        body="Resolves YES if the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025. Resolves NO otherwise.",
        resolution_date=datetime(2025, 1, 1),
        question_type="binary"
    )
    
    sample_question_2 = ForecastingQuestion(
        id=uuid4(),
        title="Will the price of Ethereum be above $10,000 on 1st January 2025?",
        body="Resolves YES if the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. Resolves NO otherwise.",
        resolution_date=datetime(2025, 1, 1),
        question_type="binary"
    )
    
    sample_output = And.OutputFormat(
        P_and_Q=ForecastingQuestion(
            id=uuid4(),
            title="Will the prices of Bitcoin and Ethereum exceed $100,000 and $10,000 respectively on 1st January 2025?",
            body="Resolves YES if both of these events happen: a) the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025 AND b) the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. Resolves NO otherwise.",
            resolution_date=datetime(2025, 1, 1),
            question_type="binary"
        )
    )
    
    base_sentences = And.BaseSentenceFormat(P=sample_question_1, Q=sample_question_2)
    
    result = await and_instance.verify(sample_output, base_sentences)
    
    assert result.valid == True, f"Verification failed. Reasoning: {result.reasoning}"
    assert len(result.reasoning) > 0, "Verification reasoning should not be empty"

    result_sync = and_instance.verify_sync(sample_output, base_sentences)

    assert result_sync.valid == True, f"Verification failed. Reasoning: {result.reasoning}, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning should not be empty, sync version"

@pytest.mark.asyncio
@pytest.mark.skipif(skip_and_verify_fail, reason="Skipping AND verify test")
async def test_and_verify_fail():
    and_instance = And()
    
    sample_question_1 = ForecastingQuestion(
        id=uuid4(),
        title="Will the price of Bitcoin be above $100,000 on 1st January 2025?",
        body="Resolves YES if the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025. Resolves NO otherwise.",
        resolution_date=datetime(2025, 1, 1),
        question_type="binary"
    )
    
    sample_question_2 = ForecastingQuestion(
        id=uuid4(),
        title="Will the price of Ethereum be above $10,000 on 1st January 2025?",
        body="Resolves YES if the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. Resolves NO otherwise.",
        resolution_date=datetime(2025, 1, 1),
        question_type="binary"
    )
    
    incorrect_output = And.OutputFormat(
        P_and_Q=ForecastingQuestion(
            id=uuid4(),
            title="Will the price of Bitcoin or Ethereum exceed $100,000 or $10,000 respectively on 1st January 2025?",
            body="Resolves YES if either of these events happen: a) the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025 OR b) the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. Resolves NO otherwise.",
            resolution_date=datetime(2025, 1, 1),
            question_type="binary"
        )
    )
    
    base_sentences = And.BaseSentenceFormat(P=sample_question_1, Q=sample_question_2)
    
    result = await and_instance.verify(incorrect_output, base_sentences)
    
    assert result.valid == False, "Verification should fail for incorrect AND combination"
    assert len(result.reasoning) > 0, "Verification reasoning for failure should not be empty"

    result_sync = and_instance.verify_sync(incorrect_output, base_sentences)

    assert result_sync.valid == False, "Verification should fail for incorrect AND combination, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning for failure should not be empty, sync version"


@pytest.mark.asyncio
@pytest.mark.skipif(skip_or_verify_pass, reason="Skipping OR verify test")
async def test_or_verify_pass():
    or_instance = Or()
    
    sample_question_1 = ForecastingQuestion(
        id=uuid4(),
        title="Will the price of Bitcoin be above $100,000 on 1st January 2025?",
        body="Resolves YES if the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025. Resolves NO otherwise.",
        resolution_date=datetime(2025, 1, 1),
        question_type="binary"
    )
    
    sample_question_2 = ForecastingQuestion(
        id=uuid4(),
        title="Will the price of Ethereum be above $10,000 on 1st January 2025?",
        body="Resolves YES if the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. Resolves NO otherwise.",
        resolution_date=datetime(2025, 1, 1),
        question_type="binary"
    )
    
    sample_output = Or.OutputFormat(
        P_or_Q=ForecastingQuestion(
            id=uuid4(),
            title="Will the price of Bitcoin exceed $100,000 or Ethereum exceed $10,000 on 1st January 2025?",
            body="Resolves YES if either of these events happen: a) the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025 OR b) the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. Resolves NO if neither event occurs.",
            resolution_date=datetime(2025, 1, 1),
            question_type="binary"
        )
    )
    
    base_sentences = Or.BaseSentenceFormat(P=sample_question_1, Q=sample_question_2)
    
    result = await or_instance.verify(sample_output, base_sentences)
    
    assert result.valid == True, f"Verification failed. Reasoning: {result.reasoning}"
    assert len(result.reasoning) > 0, "Verification reasoning should not be empty"

    result_sync = or_instance.verify_sync(sample_output, base_sentences)

    assert result_sync.valid == True, f"Verification failed. Reasoning: {result.reasoning}, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning should not be empty, sync version"

@pytest.mark.asyncio
@pytest.mark.skipif(skip_or_verify_pass, reason="Skipping OR verify test")
async def test_or_verify_fail():
    or_instance = Or()
    
    sample_question_1 = ForecastingQuestion(
        id=uuid4(),
        title="Will the price of Bitcoin be above $100,000 on 1st January 2025?",
        body="Resolves YES if the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025. Resolves NO otherwise.",
        resolution_date=datetime(2025, 1, 1),
        question_type="binary"
    )
    
    sample_question_2 = ForecastingQuestion(
        id=uuid4(),
        title="Will the price of Ethereum be above $10,000 on 1st January 2025?",
        body="Resolves YES if the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. Resolves NO otherwise.",
        resolution_date=datetime(2025, 1, 1),
        question_type="binary"
    )
    
    incorrect_output = Or.OutputFormat(
        P_or_Q=ForecastingQuestion(
            id=uuid4(),
            title="Will the prices of Bitcoin and Ethereum both exceed $100,000 and $10,000 respectively on 1st January 2025?",
            body="Resolves YES if both of these events happen: a) the spot price of Bitcoin against USD is more than 100,000 on 1st January 2025 AND b) the spot price of Ethereum against USD is more than 10,000 on 1st January 2025. Resolves NO otherwise.",
            resolution_date=datetime(2025, 1, 1),
            question_type="binary"
        )
    )
    
    base_sentences = Or.BaseSentenceFormat(P=sample_question_1, Q=sample_question_2)
    
    result = await or_instance.verify(incorrect_output, base_sentences)
    
    assert result.valid == False, "Verification should fail for incorrect OR combination"
    assert len(result.reasoning) > 0, "Verification reasoning for failure should not be empty"

    result_sync = or_instance.verify_sync(incorrect_output, base_sentences)

    assert result_sync.valid == False, "Verification should fail for incorrect OR combination, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning for failure should not be empty, sync version"


@pytest.mark.asyncio
@pytest.mark.skipif(skip_paraphrase_verify_pass, reason="Skipping paraphrase verify test")
async def test_paraphrase_verify_pass():
    paraphrase_instance = Paraphrase()
    
    sample_question = ForecastingQuestion(
        id=uuid4(),
        title="Will SpaceX successfully land humans on Mars before 2030?",
        body="Resolves YES if SpaceX achieves a successful crewed landing on the surface of Mars before January 1, 2030. The landing must be officially confirmed by SpaceX and independent space agencies. Resolves NO otherwise or if the deadline passes without such a landing.",
        resolution_date=datetime(2030, 1, 1),
        question_type="binary"
    )
    
    sample_output = Paraphrase.OutputFormat(
        para_P=ForecastingQuestion(
            id=uuid4(),
            title="Is a human Mars landing by SpaceX going to occur prior to 2030?",
            body="This question resolves positively if, before the start of 2030, SpaceX accomplishes a verified human touchdown on Mars. Verification must come from both SpaceX and independent space organizations. It resolves negatively if this doesn't happen by the specified date.",
            resolution_date=datetime(2030, 1, 1),
            question_type="binary"
        )
    )
    
    base_sentences = Paraphrase.BaseSentenceFormat(P=sample_question)
    
    result = await paraphrase_instance.verify(sample_output, base_sentences)
    
    assert result.valid == True, f"Verification failed. Reasoning: {result.reasoning}"
    assert len(result.reasoning) > 0, "Verification reasoning should not be empty"
    assert paraphrase_instance.verify_length(sample_output, base_sentences), "Paraphrase length verification failed"

    result_sync = paraphrase_instance.verify_sync(sample_output, base_sentences)

    assert result_sync.valid == True, f"Verification failed. Reasoning: {result.reasoning}, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning should not be empty, sync version"
    assert paraphrase_instance.verify_length(sample_output, base_sentences), "Paraphrase length verification failed, sync version"

@pytest.mark.asyncio
@pytest.mark.skipif(skip_paraphrase_verify_fail, reason="Skipping paraphrase verify test")
async def test_paraphrase_verify_fail():
    paraphrase_instance = Paraphrase()
    
    sample_question = ForecastingQuestion(
        id=uuid4(),
        title="Will the global average temperature increase by more than 1.5°C above pre-industrial levels by 2050?",
        body="Resolves YES if the global mean surface temperature, as reported by the World Meteorological Organization, exceeds 1.5°C above the 1850-1900 average by December 31, 2050. Resolves NO if this threshold is not reached by this date.",
        resolution_date=datetime(2050, 12, 31),
        question_type="binary"
    )
    
    incorrect_output = Paraphrase.OutputFormat(
        para_P=ForecastingQuestion(
            id=uuid4(),
            title="Is it probable that global warming will surpass 1.5°C by 2050?",
            body="Resolves YES if climate models and expert consensus indicate a high likelihood of global average temperature rising more than 1.5°C above pre-industrial levels by 2050. Resolves NO if models and experts suggest this is unlikely.",
            resolution_date=datetime(2050, 12, 31),
            question_type="binary"
        )
    )
    
    base_sentences = Paraphrase.BaseSentenceFormat(P=sample_question)
    
    result = await paraphrase_instance.verify(incorrect_output, base_sentences)
    
    assert result.valid == False, "Verification should fail for incorrect paraphrase"
    assert len(result.reasoning) > 0, "Verification reasoning for failure should not be empty"
    assert paraphrase_instance.verify_length(incorrect_output, base_sentences), "Paraphrase length verification failed"

    result_sync = paraphrase_instance.verify_sync(incorrect_output, base_sentences)

    assert result_sync.valid == False, "Verification should fail for incorrect paraphrase, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning for failure should not be empty, sync version"
    assert paraphrase_instance.verify_length(incorrect_output, base_sentences), "Paraphrase length verification failed, sync version"

@pytest.mark.asyncio
@pytest.mark.skipif(skip_conditional_verify_pass, reason="Skipping conditional verify test")
async def test_conditional_verify_pass():
    conditional_instance = Conditional()
    
    sample_question_P = ForecastingQuestion(
        id=uuid4(),
        title="Will the US Federal Reserve raise interest rates in Q3 2024?",
        body="Resolves YES if the US Federal Reserve increases the federal funds rate at any of its meetings in July, August, or September 2024. Resolves NO if rates remain unchanged or are lowered during this period.",
        resolution_date=datetime(2024, 9, 30),
        question_type="binary"
    )
    
    sample_question_Q = ForecastingQuestion(
        id=uuid4(),
        title="Will US inflation rate drop below 2% by the end of 2024?",
        body="Resolves YES if the US Consumer Price Index (CPI) shows an annual inflation rate below 2% for December 2024, as reported by the US Bureau of Labor Statistics. Resolves NO if the inflation rate is 2% or higher.",
        resolution_date=datetime(2024, 12, 31),
        question_type="binary"
    )
    
    sample_output = Conditional.OutputFormat(
        Q_given_P=ForecastingQuestion(
            id=uuid4(),
            title="Given that the US Federal Reserve raises interest rates in Q3 2024, will US inflation rate drop below 2% by the end of 2024?",
            body="This question resolves YES if both of the following conditions are met:\n1. The US Federal Reserve increases the federal funds rate at any of its meetings in July, August, or September 2024, AND\n2. The US Consumer Price Index (CPI) shows an annual inflation rate below 2% for December 2024, as reported by the US Bureau of Labor Statistics.\nResolves NO if condition 1 is met but condition 2 is not.\nResolves N/A if the Federal Reserve does not raise interest rates in Q3 2024.",
            resolution_date=datetime(2024, 12, 31),
            question_type="conditional_binary"
        )
    )
    
    base_sentences = Conditional.BaseSentenceFormat(P=sample_question_P, Q=sample_question_Q)
    
    result = await conditional_instance.verify(sample_output, base_sentences)
    
    assert result.valid == True, f"Verification failed. Reasoning: {result.reasoning}"
    assert len(result.reasoning) > 0, "Verification reasoning should not be empty"
    assert conditional_instance.verify_length(sample_output, base_sentences), "Conditional length verification failed"

    result_sync = conditional_instance.verify_sync(sample_output, base_sentences)

    assert result_sync.valid == True, f"Verification failed. Reasoning: {result.reasoning}, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning should not be empty, sync version"
    assert conditional_instance.verify_length(sample_output, base_sentences), "Conditional length verification failed, sync version"

@pytest.mark.asyncio
@pytest.mark.skipif(skip_conditional_verify_fail, reason="Skipping conditional verify test")
async def test_conditional_verify_fail():
    conditional_instance = Conditional()
    
    sample_question_P = ForecastingQuestion(
        id=uuid4(),
        title="Will SpaceX launch Starship successfully to orbit in 2024?",
        body="Resolves YES if SpaceX successfully launches Starship to orbit and completes at least one full orbit around Earth in 2024. The launch must be officially confirmed by SpaceX and verified by independent space agencies.",
        resolution_date=datetime(2024, 12, 31),
        question_type="binary"
    )
    
    sample_question_Q = ForecastingQuestion(
        id=uuid4(),
        title="Will NASA's Artemis II mission launch in 2024?",
        body="Resolves YES if NASA launches the Artemis II crewed mission around the Moon in 2024. The launch must be officially confirmed by NASA and the mission must successfully orbit the Moon.",
        resolution_date=datetime(2024, 12, 31),
        question_type="binary"
    )
    
    incorrect_output = Conditional.OutputFormat(
        Q_given_P=ForecastingQuestion(
            id=uuid4(),
            title="If SpaceX launches Starship to orbit in 2024, how likely is NASA to launch Artemis II in the same year?",
            body="Resolves YES if SpaceX successfully launches Starship to orbit and completes at least one full orbit around Earth in 2024. The launch must be officially confirmed by SpaceX and verified by independent space agencies. Resolves NO if NASA launches the Artemis II crewed mission around the Moon in 2024. The launch must be officially confirmed by NASA and the mission must successfully orbit the Moon.",
            resolution_date=datetime(2024, 12, 31),
            question_type="conditional_binary"
        )
    )
    
    base_sentences = Conditional.BaseSentenceFormat(P=sample_question_P, Q=sample_question_Q)
    
    result = await conditional_instance.verify(incorrect_output, base_sentences)
    
    assert result.valid == False, "Verification should fail for incorrect conditional"
    assert len(result.reasoning) > 0, "Verification reasoning for failure should not be empty"
    assert conditional_instance.verify_length(incorrect_output, base_sentences), "Conditional length verification failed"

    result_sync = conditional_instance.verify_sync(incorrect_output, base_sentences)

    assert result_sync.valid == False, "Verification should fail for incorrect conditional, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning for failure should not be empty, sync version"
    assert conditional_instance.verify_length(incorrect_output, base_sentences), "Conditional length verification failed, sync version"


@pytest.mark.asyncio
@pytest.mark.skipif(skip_consequence_quantity_verify_pass, reason="Skipping consequence quantity verify test")
async def test_consequence_quantity_verify_pass():
    consequence_instance = Consequence()
    
    sample_question_P = ForecastingQuestion(
        id=uuid4(),
        title="Will the global smartphone market exceed 1.5 billion units sold in 2026?",
        body="Resolves YES if the total number of smartphones sold worldwide in the year 2026 surpasses 1.5 billion units, as reported by IDC, Gartner, or another reputable technology market research firm. Resolves NO otherwise.",
        resolution_date=datetime(2026, 12, 31),
        question_type="binary"
    )
    
    sample_output = Consequence.OutputFormat(
        cons_P=ForecastingQuestion(
            id=uuid4(),
            title="Will the global smartphone market exceed 1.3 billion units sold in 2026?",
            body="Resolves YES if the total number of smartphones sold worldwide in the year 2026 surpasses 1.3 billion units, as reported by IDC, Gartner, or another reputable technology market research firm. Resolves NO otherwise.",
            resolution_date=datetime(2026, 12, 31),
            question_type="binary",
            metadata={"consequence_type": "quantity"}
        )
    )
    
    base_sentences = Consequence.BaseSentenceFormat(P=sample_question_P)
    
    result = await consequence_instance.verify(sample_output, base_sentences)
    
    assert result.valid == True, f"Quantity verification failed. Reasoning: {result.reasoning}"
    assert len(result.reasoning) > 0, "Verification reasoning should not be empty"
    assert consequence_instance.verify_length(sample_output, base_sentences), "Consequence length verification failed"

    result_sync = consequence_instance.verify_sync(sample_output, base_sentences)

    assert result_sync.valid == True, f"Quantity verification failed. Reasoning: {result.reasoning}, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning should not be empty, sync version"
    assert consequence_instance.verify_length(sample_output, base_sentences), "Consequence length verification failed, sync version"

@pytest.mark.asyncio
@pytest.mark.skipif(skip_consequence_quantity_verify_fail, reason="Skipping consequence quantity verify test")
async def test_consequence_time_verify_pass():
    consequence_instance = Consequence()
    
    sample_question_P = ForecastingQuestion(
        id=uuid4(),
        title="Will quantum computers achieve quantum supremacy for a practical problem by 2028?",
        body="Resolves YES if a quantum computer demonstrably solves a practical problem faster than any classical computer before January 1, 2028. The achievement must be published in a peer-reviewed scientific journal and acknowledged by the scientific community. Resolves NO otherwise.",
        resolution_date=datetime(2028, 1, 1),
        question_type="binary"
    )
    
    sample_output = Consequence.OutputFormat(
        cons_P=ForecastingQuestion(
            id=uuid4(),
            title="Will quantum computers achieve quantum supremacy for a practical problem by 2030?",
            body="Resolves YES if a quantum computer demonstrably solves a practical problem faster than any classical computer before January 1, 2030. The achievement must be published in a peer-reviewed scientific journal and acknowledged by the scientific community. Resolves NO otherwise.",
            resolution_date=datetime(2030, 1, 1),
            question_type="binary",
            metadata={"consequence_type": "time"}
        )
    )
    
    base_sentences = Consequence.BaseSentenceFormat(P=sample_question_P)
    
    result = await consequence_instance.verify(sample_output, base_sentences)
    
    assert result.valid == True, f"Time verification failed. Reasoning: {result.reasoning}"
    assert len(result.reasoning) > 0, "Verification reasoning should not be empty"
    assert consequence_instance.verify_length(sample_output, base_sentences), "Consequence length verification failed"

    result_sync = consequence_instance.verify_sync(sample_output, base_sentences)

    assert result_sync.valid == True, f"Time verification failed. Reasoning: {result.reasoning}, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning should not be empty, sync version"
    assert consequence_instance.verify_length(sample_output, base_sentences), "Consequence length verification failed, sync version"


@pytest.mark.asyncio
@pytest.mark.skipif(skip_consequence_time_verify_fail, reason="Skipping consequence time verify test")
async def test_consequence_misc_verify_pass():
    consequence_instance = Consequence()
    
    sample_question_P = ForecastingQuestion(
        id=uuid4(),
        title="Will a private company successfully remove 1000 pieces of space debris from Earth's orbit by 2035?",
        body="Resolves YES if a private company (not government-owned or operated) successfully removes at least 1000 pieces of space debris (defined as non-functional human-made objects in Earth's orbit) from orbit by December 31, 2035. The removal must be verified by independent space agencies or reputable space tracking organizations. Resolves NO otherwise.",
        resolution_date=datetime(2035, 12, 31),
        question_type="binary"
    )
    
    sample_output = Consequence.OutputFormat(
        cons_P=ForecastingQuestion(
            id=uuid4(),
            title="Will at least 500 pieces of space debris be removed from Earth's orbit by 2035?",
            body="Resolves YES if at least 500 pieces of space debris (defined as non-functional human-made objects in Earth's orbit) are removed from orbit by any entity or combination of entities (including private companies, government agencies, or international collaborations) by December 31, 2035. The removal must be verified by independent space agencies or reputable space tracking organizations. Resolves NO otherwise.",
            resolution_date=datetime(2035, 12, 31),
            question_type="binary",
            metadata={"consequence_type": "misc"}
        )
    )
    
    base_sentences = Consequence.BaseSentenceFormat(P=sample_question_P)
    
    result = await consequence_instance.verify(sample_output, base_sentences)
    
    assert result.valid == True, f"Misc verification failed. Reasoning: {result.reasoning}"
    assert len(result.reasoning) > 0, "Verification reasoning should not be empty"
    assert consequence_instance.verify_length(sample_output, base_sentences), "Consequence length verification failed"

    result_sync = consequence_instance.verify_sync(sample_output, base_sentences)

    assert result_sync.valid == True, f"Misc verification failed. Reasoning: {result.reasoning}, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning should not be empty, sync version"
    assert consequence_instance.verify_length(sample_output, base_sentences), "Consequence length verification failed, sync version"

@pytest.mark.asyncio
@pytest.mark.skipif(skip_consequence_misc_verify_fail, reason="Skipping consequence misc verify test")
async def test_consequence_misc_verify_fail():
    consequence_instance = Consequence()
    
    sample_question_P = ForecastingQuestion(
        id=uuid4(),
        title="Will a multinational company successfully develop a commercially viable fusion reactor by 2040?",
        body="Resolves YES if a multinational company (defined as a company operating in more than one country) successfully develops and demonstrates a fusion reactor that produces more energy than it consumes, and announces plans for commercial deployment, by December 31, 2040. The achievement must be independently verified by reputable scientific institutions and acknowledged by the International Atomic Energy Agency or a similar international body. Resolves NO otherwise.",
        resolution_date=datetime(2040, 12, 31),
        question_type="binary"
    )
    
    incorrect_output = Consequence.OutputFormat(
        cons_P=ForecastingQuestion(
            id=uuid4(),
            title="Will global electricity prices decrease by at least 20% by 2041?",
            body="Resolves YES if the average global electricity price, as reported by the International Energy Agency or a similar reputable source, decreases by 20% or more from 2023 levels by December 31, 2041. The decrease must be sustained for at least six months to account for short-term fluctuations. Resolves NO otherwise.",
            resolution_date=datetime(2041, 12, 31),
            question_type="binary",
            metadata={"consequence_type": "misc"}
        )
    )
    
    base_sentences = Consequence.BaseSentenceFormat(P=sample_question_P)
    
    result = await consequence_instance.verify(incorrect_output, base_sentences)
    
    assert result.valid == False, "Misc verification should fail for incorrect consequence"
    assert len(result.reasoning) > 0, "Verification reasoning for failure should not be empty"
    assert consequence_instance.verify_length(incorrect_output, base_sentences), "Consequence length verification failed"

    result_sync = consequence_instance.verify_sync(incorrect_output, base_sentences)

    assert result_sync.valid == False, "Misc verification should fail for incorrect consequence, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning for failure should not be empty, sync version"
    assert consequence_instance.verify_length(incorrect_output, base_sentences), "Consequence length verification failed, sync version"

@pytest.mark.asyncio
@pytest.mark.skipif(skip_consequence_quantity_verify_fail, reason="Skipping consequence quantity verify test")
async def test_consequence_quantity_verify_fail():
    consequence_instance = Consequence()
    
    sample_question_P = ForecastingQuestion(
        id=uuid4(),
        title="Will the global smartphone market exceed 1.5 billion units sold in 2026?",
        body="Resolves YES if the total number of smartphones sold worldwide in the year 2026 surpasses 1.5 billion units, as reported by IDC, Gartner, or another reputable technology market research firm. Resolves NO otherwise.",
        resolution_date=datetime(2026, 12, 31),
        question_type="binary"
    )
    
    incorrect_output = Consequence.OutputFormat(
        cons_P=ForecastingQuestion(
            id=uuid4(),
            title="Will the global smartphone market exceed 1.7 billion units sold in 2026?",
            body="Resolves YES if the total number of smartphones sold worldwide in the year 2026 surpasses 1.7 billion units, as reported by IDC, Gartner, or another reputable technology market research firm. Resolves NO otherwise.",
            resolution_date=datetime(2026, 12, 31),
            question_type="binary",
            metadata={"consequence_type": "quantity"}
        )
    )
    
    base_sentences = Consequence.BaseSentenceFormat(P=sample_question_P)
    
    result = await consequence_instance.verify(incorrect_output, base_sentences)
    
    assert result.valid == False, "Quantity verification should fail for incorrect consequence"
    assert len(result.reasoning) > 0, "Verification reasoning for failure should not be empty"
    assert consequence_instance.verify_length(incorrect_output, base_sentences), "Consequence length verification failed"

    result_sync = consequence_instance.verify_sync(incorrect_output, base_sentences)

    assert result_sync.valid == False, "Quantity verification should fail for incorrect consequence, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning for failure should not be empty, sync version"
    assert consequence_instance.verify_length(incorrect_output, base_sentences), "Consequence length verification failed, sync version"


@pytest.mark.asyncio
@pytest.mark.skipif(skip_consequence_time_verify_fail, reason="Skipping consequence time verify test")
async def test_consequence_time_verify_fail():
    consequence_instance = Consequence()
    
    sample_question_P = ForecastingQuestion(
        id=uuid4(),
        title="Will a human land on Europa before 2050?",
        body="Resolves YES if a human astronaut successfully lands on the surface of Jupiter's moon Europa before January 1, 2050. The landing must be verified by multiple space agencies and published in peer-reviewed scientific journals. Resolves NO otherwise.",
        resolution_date=datetime(2050, 1, 1),
        question_type="binary"
    )
    
    incorrect_output = Consequence.OutputFormat(
        cons_P=ForecastingQuestion(
            id=uuid4(),
            title="Will a human land on Europa before 2045?",
            body="Resolves YES if a human astronaut successfully lands on the surface of Jupiter's moon Europa before January 1, 2045. The landing must be verified by multiple space agencies and published in peer-reviewed scientific journals. Resolves NO otherwise.",
            resolution_date=datetime(2045, 1, 1),
            question_type="binary",
            metadata={"consequence_type": "time"}
        )
    )
    
    base_sentences = Consequence.BaseSentenceFormat(P=sample_question_P)
    
    result = await consequence_instance.verify(incorrect_output, base_sentences)
    
    assert result.valid == False, "Time verification should fail for incorrect consequence"
    assert len(result.reasoning) > 0, "Verification reasoning for failure should not be empty"
    assert consequence_instance.verify_length(incorrect_output, base_sentences), "Consequence length verification failed"

    result_sync = consequence_instance.verify_sync(incorrect_output, base_sentences)

    assert result_sync.valid == False, "Time verification should fail for incorrect consequence, sync version"
    assert len(result_sync.reasoning) > 0, "Verification reasoning for failure should not be empty, sync version"
    assert consequence_instance.verify_length(incorrect_output, base_sentences), "Consequence length verification failed, sync version"
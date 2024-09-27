from common.datatypes import (
    ForecastingQuestion,
    ForecastingQuestion_stripped,
    RelevanceResult,
)
from common.llm_utils import answer_sync, answer, Example, parallelized_call
from common.path_utils import get_data_path
from pathlib import Path
import jsonlines
import random
import functools

MODEL_RELEVANCE = "gpt-4o-mini-2024-07-18"
BASE_DATA_PATH: Path = (
    get_data_path() / "fq" / "real" / "questions_cleaned_formatted.jsonl"
)
# BASE_DATA_PATH: Path = (
#    get_data_path() / "fq" / "synthetic" / "high-quality-questions--all-domains.jsonl"
# )


preface = r"""I'm doing a project that involve eliciting probabilities from LLMs to measure the calibration, consistency 
and such properties of LLM forecasters. As part of this project we will be taking logical combinations of forecasting 
questions and eliciting probabilities on them. I need your help in deciding, for two given forecasting questions, 
whether it makes sense to think about their logical combinations/whether it's worth doing so.

For example, we might want to elicit the probability of

'Will Donald Trump win the 2024 US presidential election? AND Will US economic growth exceed 3.5% in 2025?'

because Trump winning the election might potentially (positively or negatively) affect economic growth in the following year.

But we probably wouldn't care about the probability of 

'Will Donald Trump win the 2024 US presidential election? AND Will the men's deadlift record be broken in 2025?'

because those seem wholly unrelated.

Can you help me with this? I will just give you two forecasting questions, and you must give me 

1) One or more examples of reasons someone might be interested in the logical combination of those questions; based on how 
realistic these reason(s) are, provide--

2) a score between 0 and 10 to advise me on whether it makes sense to consider their logical combination 
(with 0 being 'the logical combination is nonsensical, nobody would ever ask something like that', 10 being 
'yeah that's a perfectly legitimate question I could imagine seeing that on Manifold or Metaculus')
"""


def simple_combine(
    *base_questions: list[ForecastingQuestion_stripped],
) -> ForecastingQuestion_stripped:
    return ForecastingQuestion_stripped(
        title=" AND ".join(bq.title for bq in base_questions),
        body="Resolution criteria for each question:\n\n"
        + "\n".join(bq.body for bq in base_questions),
    )


examples = [
    Example(
        user=simple_combine(
            ForecastingQuestion_stripped(
                title="Will Donald Trump win the 2024 US presidential election?",
                body="",
            ),
            ForecastingQuestion_stripped(
                title=r"Will US economic growth exceed 3.5% in 2023?",
                body="",
            ),
        ),
        assistant=RelevanceResult(
            reasons=[
                (
                    "**Political and economic policy impact**:\n"
                    " - Trump's policy agenda could influence market confidence and economic performance. Historically, Trump's administration focused on deregulation, tax cuts, and trade policies that could have short-term impacts on economic growth.\n"
                    " - Analysts and investors might be interested in understanding how the prospect of Trump's victory affects economic forecasts and vice versa."
                ),
                (
                    "**Market reactions**:\n"
                    "The anticipation of Trump's victory could influence market behavior, including stock market performance and investment trends, which in turn could affect economic growth."
                ),
                (
                    "**Predictive indicators**:\n"
                    "Some economic models or forecasters might use political outcomes as inputs. Understanding the combined probability could improve the calibration of these models."
                ),
            ],
            conclusion=(
                "This combination is quite reasonable. There is a clear and plausible link between the outcome of a major election and "
                "subsequent economic performance. While the election in question is for 2024, and the economic growth measure is for "
                "2023, forecasters might be interested in how expectations about the 2024 election might be shaping economic behavior "
                "and predictions in 2023. Conversely, the election result may depend on economic growth in the previous year (e.g. "
                "anti-incumbency sentiment due to a recession). Therefore, this logical combination makes sense and could be of "
                "interest to a range of stakeholders from political analysts to economic forecasters."
            ),
            score=8.0,
        ),
    ),
    Example(
        user=simple_combine(
            ForecastingQuestion_stripped(
                title="Will there be Human-machine intelligence parity before 2040?",
                body="",
            ),
            ForecastingQuestion_stripped(
                title=r"Will there be a Vegan Country by 2100?",
                body="",
            ),
        ),
        assistant=RelevanceResult(
            reasons=[
                (
                    "**Technological and societal transformation**:\n"
                    " - Achieving human-machine intelligence parity (HMI parity) could lead to significant societal changes, potentially accelerating advancements in various fields, including ethics, agriculture, and environmental policies.\n"
                    " - The rise of advanced AI might influence global trends toward sustainability and ethical living, possibly supporting the movement towards a vegan country."
                ),
                (
                    "**Ethical and Cultural Shifts:**:\n"
                    "With HMI parity, AI could play a crucial role in promoting ethical consumption and sustainability through advanced education, advocacy, and policy-making, potentially increasing the likelihood of a country adopting veganism."
                ),
                (
                    "**Predictive Modeling and Futurism:**:\n"
                    "Futurists and analysts might be interested in exploring how technological advancements could correlate with or influence major societal changes. This could include the adoption of veganism at a national level, driven by AI’s potential to optimize resource use and promote ethical decisions."
                ),
                (
                    "**Policy and Governance**:\n"
                    "Advanced AI systems could aid governments in making more informed and ethical decisions, potentially supporting policies that lead to a nation becoming vegan."
                ),
            ],
            conclusion=(
                "While there is some speculative connection between technological advancements (like HMI parity) and major societal "
                "changes (like the establishment of a vegan country), the direct link is less immediate compared to more directly "
                "related questions. However, it is plausible and interesting enough for futurists and those studying long-term "
                "societal impacts of technology. The combination is not entirely out of the realm of possibility, but it’s less "
                "directly related compared to more immediate socio-economic issues."
            ),
            score=6.0,
        ),
    ),
    Example(
        user=simple_combine(
            ForecastingQuestion_stripped(
                title="Will RoboCup announce that robots have beaten professional human soccer players before 2050?",
                body="",
            ),
            ForecastingQuestion_stripped(
                title=r"Will there be more VR headset sales than AR headsets sales in 2025?",
                body="",
            ),
        ),
        assistant=RelevanceResult(
            reasons=[
                (
                    "**Technological Advancement Trends**:\n"
                    "Both questions involve significant milestones in technology and could reflect broader trends in AI and consumer tech development. Comparing them might give insights into the pace and direction of technological advancements in different fields.\n"
                ),
                (
                    "**Market and Industry Dynamics**:\n"
                    "Success in competitive robotics (such as RoboCup robots beating professional soccer players) might correlate with or influence trends in consumer tech adoption. Advanced robotics could drive interest and innovation in immersive technologies like VR and AR."
                ),
                (
                    "**Public Interest and Investment**:\n"
                    "Achievements in robotics could impact public interest and investment in related technologies, including VR and AR. For instance, breakthroughs in robotics could spur interest in immersive sports experiences, potentially influencing headset sales."
                ),
                (
                    "**Synergistic Technological Ecosystems**:\n"
                    "Advances in AI and robotics could complement and enhance VR and AR technologies. Understanding how different tech fields progress together might help in forecasting future trends and market behaviors."
                ),
            ],
            conclusion=(
                "This combination makes a reasonable amount of sense. While the direct connection between robots beating human "
                "soccer players and VR vs. AR headset sales is not immediately obvious, both involve significant technological "
                "milestones and market trends. Analysts and tech forecasters might be interested in understanding how advancements "
                "in one area of AI and robotics could correlate with or impact trends in another area of consumer technology. "
                "Therefore, this logical combination is plausible and could be of interest to those tracking technological progress "
                "and market dynamics."
            ),
            score=7.0,
        ),
    ),
    Example(
        user=simple_combine(
            ForecastingQuestion_stripped(
                title="Will any of Belgium, France, Italy, Luxembourg, Netherlands, and/or Germany leave the EU before 2027?",
                body="",
            ),
            ForecastingQuestion_stripped(
                title=r"Will we reach the island of stability by 2050?",
                body='Resolution Criteria\nSince the synthesis of neptunium in 1940, we have been continually expanding the periodic table by creating new elements. Regrettably, as atoms have become bigger, they also have become less stable, the last few elements to be created having a half-life of less than a second.\nYet it is theorized that at some point, stability of new elements might start increasing again, creating an island of stability. There are certain "magic numbers" of protons that offer the chance of higher stability; 114, 120 and 126 are magic numbers. We have yet to reach elements 120 and 126 and there might still be more stable isotopes of element 114 that have not yet been created.\nIt is asked:\nWill we create an isotope of an element that has more than 110 protons, that has a half-life of at least one day (86,400 seconds) prior to 2050?\nIn order for the question to resolve positive the half-life of the isotope must be verified by an independent scientific team to be greater than one day prior to 2050.\n',
            ),
        ),
        assistant=RelevanceResult(
            reasons=[
                (
                    "**Geopolitical and Scientific Milestones**:\n"
                    "Both questions deal with significant milestones in their respective fields: geopolitical changes in the EU and advancements in nuclear physics. Observing these milestones together could provide a broader perspective on human progress and societal stability.\n"
                ),
                (
                    "**Impact of Geopolitical Stability on Scientific Research**:\n"
                    "The stability and collaboration within the EU can significantly impact scientific research funding and international collaborations. A major geopolitical shift, such as a country leaving the EU, could influence scientific priorities and resources, potentially affecting long-term projects like achieving the island of stability."
                ),
                (
                    "**Public and Policy Interest**:\n"
                    "Policymakers and the public might be interested in how major geopolitical changes could correlate with scientific advancements. Understanding the interplay between political stability and scientific progress could help in forecasting future trends and setting policies."
                ),
                (
                    "**Scenario Planning**:\n"
                    "Futurists and analysts might use such combinations to create scenarios that explore the relationship between political upheaval and scientific breakthroughs, helping to anticipate and plan for future developments."
                ),
            ],
            conclusion=(
                "This combination has some speculative connections but is less directly related compared to more immediate and obvious "
                "pairs. While there is an interesting angle in considering how geopolitical stability might influence scientific progress, "
                "the direct impact of one on the other is less clear. However, the combination can still be of interest for broader "
                "scenario planning and understanding the interplay between political and scientific developments."
            ),
            score=5.0,
        ),
    ),
    Example(
        user=simple_combine(
            ForecastingQuestion_stripped(
                title="Will it rain in Paris on July 4th, 2025?",
                body="",
            ),
            ForecastingQuestion_stripped(
                title=r"Will a new species of deep-sea fish be discovered in 2025",
                body="",
            ),
        ),
        assistant=RelevanceResult(
            reasons=[
                (
                    "**Scientific Curiosity**:\n"
                    "Researchers interested in broad environmental patterns might explore how different kinds of natural phenomena (like weather patterns and biological discoveries) occur within the same timeframe. This could help in understanding broader trends in nature and scientific discovery."
                ),
                (
                    "**Exploratory Studies**:\n"
                    "Some studies may take a broad exploratory approach, looking for correlations or patterns in seemingly unrelated events. This could be part of a larger data-mining effort to identify unexpected connections."
                ),
                (
                    "**Public Engagement**:\n"
                    "Such combinations might be used in public science communication or educational settings to engage people with diverse scientific topics, fostering curiosity about both meteorology and marine biology."
                ),
                (
                    "**Scenario Planning**:\n"
                    "Futurists and analysts might use such combinations to create scenarios that explore the relationship between political upheaval and scientific breakthroughs, helping to anticipate and plan for future developments."
                ),
            ],
            conclusion=(
                "This combination is largely nonsensical as the two events are highly unrelated and do not have any "
                "meaningful correlation. The weather in Paris on a specific day and the discovery of a new species "
                "of deep-sea fish are independent events with no logical or causal connection. It is unlikely that "
                "anyone would have a practical or scientific interest in the combined probability of these events. "
                "This pair falls into the category of interesting trivia at best but not useful for forecasting or "
                "analysis."
            ),
            score=0.0,
        ),
    ),
]


def relevance_sync(
    base_sentences: dict[str, ForecastingQuestion] | list[ForecastingQuestion],
    **kwargs,
) -> float:
    """Gives a score to assess if it's worth instantiating some given combination of base sentences."""

    if isinstance(base_sentences, dict):
        base_sentences = list(base_sentences.values())

    result = answer_sync(
        simple_combine(*base_sentences),
        preface=preface,
        # examples=examples,
        response_model=RelevanceResult,
        **kwargs,
    )

    print("---")
    print(base_sentences)
    print(result)
    print("---")

    return {"relevance": result.model_dump()}


async def relevance(
    base_sentences: dict[str, ForecastingQuestion] | list[ForecastingQuestion], **kwargs
) -> dict:
    """Gives a score to assess if it's worth instantiating some given combination of base sentences."""

    if isinstance(base_sentences, dict):
        base_sentences = list(base_sentences.values())

    result = await answer(
        simple_combine(*base_sentences),
        preface=preface,
        # examples=examples,
        response_model=RelevanceResult,
        **kwargs,
    )

    print("---")
    print(base_sentences)
    print(result)
    print("---")

    return {"relevance": result.model_dump()}


async def get_relevant_questions(
    base_data_path: Path = BASE_DATA_PATH,
    existing_questions: list[ForecastingQuestion] = None,
    n_relevance: int = 10,
    n_return: int = 1,
    tuple_size=2,
    seed: int = 42,
    **kwargs,
) -> list[tuple[list[ForecastingQuestion], dict]]:
    if existing_questions is None:
        existing_questions = []

    if len(existing_questions) >= tuple_size:
        return [
            (
                existing_questions,
                {"relevance": {"score": 10.0, "reasons": [], "conclusion": "N/A"}},
            )
            for _ in range(n_return)
        ]

    bqs = []
    print(f"Loading questions from {base_data_path}...")
    for line in jsonlines.open(base_data_path):
        try:
            bq = ForecastingQuestion(**line)
            bqs.append(bq)
        except Exception as e:
            print(e)
            continue
    print(f"Loaded {len(bqs)} questions.")

    random.seed(seed)

    sampled_qs = [
        random.sample(bqs, tuple_size - len(existing_questions))
        for _ in range(n_relevance)
    ]

    tuples = [existing_questions + list(sample) for sample in sampled_qs]

    relevances = await parallelized_call(
        functools.partial(relevance, **kwargs),
        tuples,
    )

    tuples_with_relevances = list(zip(tuples, relevances))

    tuples_with_relevances.sort(key=lambda x: x[1]["relevance"]["score"], reverse=True)

    return tuples_with_relevances[:n_return]


def get_relevant_questions_sync(
    base_data_path: Path = BASE_DATA_PATH,
    existing_questions: list[ForecastingQuestion] = None,
    n_relevance: int = 10,
    n_return: int = 1,
    tuple_size=2,
    seed: int = 42,
    **kwargs,
) -> list[tuple[list[ForecastingQuestion], dict]]:
    if existing_questions is None:
        existing_questions = []

    bqs = []
    print(f"Loading questions from {base_data_path}...")
    for line in jsonlines.open(base_data_path):
        try:
            bq = ForecastingQuestion(**line)
            bqs.append(bq)
        except Exception as e:
            print(e)
            continue
    print(f"Loaded {len(bqs)} questions.")

    random.seed(seed)

    sampled_qs = [
        random.sample(bqs, tuple_size - len(existing_questions))
        for _ in range(n_relevance)
    ]

    tuples = [existing_questions + list(sample) for sample in sampled_qs]

    relevances = [relevance_sync(base_sentences=tuple, **kwargs) for tuple in tuples]

    tuples_with_relevances = list(zip(tuples, relevances))

    tuples_with_relevances.sort(key=lambda x: x[1]["relevance"]["score"], reverse=True)

    return tuples_with_relevances[:n_return]

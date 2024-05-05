from .forecaster import Forecaster
from common.datatypes import ForecastingQuestion_stripped, ForecastingQuestion
from common.llm_utils import Example

# llm_forecasting imports
from config.constants import PROMPT_DICT
import ranking
import summarize
import ensemble

RETRIEVAL_CONFIG = {
    "NUM_SEARCH_QUERY_KEYWORDS": 3,
    "MAX_WORDS_NEWSCATCHER": 5,
    "MAX_WORDS_GNEWS": 8,
    "SEARCH_QUERY_MODEL_NAME": "gpt-4-1106-preview",
    "SEARCH_QUERY_TEMPERATURE": 0.0,
    "SEARCH_QUERY_PROMPT_TEMPLATES": [
        PROMPT_DICT["search_query"]["0"],
        PROMPT_DICT["search_query"]["1"],
    ],
    "NUM_ARTICLES_PER_QUERY": 5,
    "SUMMARIZATION_MODEL_NAME": "gpt-3.5-turbo-1106",
    "SUMMARIZATION_TEMPERATURE": 0.2,
    "SUMMARIZATION_PROMPT_TEMPLATE": PROMPT_DICT["summarization"]["9"],
    "PRE_FILTER_WITH_EMBEDDING": True,
    "PRE_FILTER_WITH_EMBEDDING_THRESHOLD": 0.32,
    "RANKING_MODEL_NAME": "gpt-3.5-turbo-1106",
    "RANKING_TEMPERATURE": 0.0,
    "RANKING_PROMPT_TEMPLATE": PROMPT_DICT["ranking"]["0"],
    "RANKING_RELEVANCE_THRESHOLD": 4,
    "RANKING_COSINE_SIMILARITY_THRESHOLD": 0.5,
    "SORT_BY": "date",
    "RANKING_METHOD": "llm-rating",
    "RANKING_METHOD_LLM": "title_250_tokens",
    "NUM_SUMMARIES_THRESHOLD": 20,
    "EXTRACT_BACKGROUND_URLS": True,
}

REASONING_CONFIG = {
    "BASE_REASONING_MODEL_NAMES": ["gpt-4-1106-preview", "gpt-4-1106-preview"],
    "BASE_REASONING_TEMPERATURE": 1.0,
    "BASE_REASONING_PROMPT_TEMPLATES": [
        [
            PROMPT_DICT["binary"]["scratch_pad"]["1"],
            PROMPT_DICT["binary"]["scratch_pad"]["2"],
        ],
        [
            PROMPT_DICT["binary"]["scratch_pad"]["new_3"],
            PROMPT_DICT["binary"]["scratch_pad"]["new_6"],
        ],
    ],
    "ALIGNMENT_MODEL_NAME": "gpt-3.5-turbo-1106",
    "ALIGNMENT_TEMPERATURE": 0,
    "ALIGNMENT_PROMPT": PROMPT_DICT["alignment"]["0"],
    "AGGREGATION_METHOD": "meta",
    "AGGREGATION_PROMPT_TEMPLATE": PROMPT_DICT["meta_reasoning"]["0"],
    "AGGREGATION_TEMPERATURE": 0.2,
    "AGGREGATION_MODEL_NAME": "gpt-4",
    "AGGREGATION_WEIGTHTS": None,
}


class AdvancedForecaster(Forecaster):
    def __init__(self, preface: str = None, examples: list = None):
        self.preface = preface or (
            "You are an informed and well-calibrated forecaster. I need you to give me "
            "your best probability estimate for the following sentence or question resolving YES. "
            "Your answer should be a float between 0 and 1, with nothing else in your response."
        )

        self.examples = examples or [
            Example(
                user=ForecastingQuestion_stripped(
                    title="Will Manhattan have a skyscraper a mile tall by 2030?",
                    body=(
                        "Resolves YES if at any point before 2030, there is at least "
                        "one building in the NYC Borough of Manhattan (based on current "
                        "geographic boundaries) that is at least a mile tall."
                    ),
                ),
                assistant=0.03,
            )
        ]

    def call(self, sentence: ForecastingQuestion, **kwargs) -> float:
        raise NotImplementedError

    async def call_async(self, sentence: ForecastingQuestion, **kwargs) -> float:
        question = sentence.title
        background_info = sentence.metadata["background_info"]
        resolution_criteria = sentence.body
        retrieval_dates = (
            "2024-03-01",
            "2024-05-04",
        )  # artificially set and fixed for now

        (
            ranked_articles,
            all_articles,
            search_queries_list_gnews,
            search_queries_list_nc,
        ) = await ranking.retrieve_summarize_and_rank_articles(
            question,
            background_info,
            resolution_criteria,
            retrieval_dates,
            urls=[],
            config=RETRIEVAL_CONFIG,
            return_intermediates=True,
        )

        all_summaries = summarize.concat_summaries(
            ranked_articles[: RETRIEVAL_CONFIG["NUM_SUMMARIES_THRESHOLD"]]
        )

        # retrieval_dates[1] currently set to today
        # data doesn't have close date currently (so set to N/A)
        today_to_close_date = [retrieval_dates[1], "N/A"]
        ensemble_dict = await ensemble.meta_reason(
            question=question,
            background_info=background_info,
            resolution_criteria=resolution_criteria,
            today_to_close_date_range=today_to_close_date,
            retrieved_info=all_summaries,
            reasoning_prompt_templates=REASONING_CONFIG[
                "BASE_REASONING_PROMPT_TEMPLATES"
            ],
            base_model_names=REASONING_CONFIG["BASE_REASONING_MODEL_NAMES"],
            base_temperature=REASONING_CONFIG["BASE_REASONING_TEMPERATURE"],
            aggregation_method=REASONING_CONFIG["AGGREGATION_METHOD"],
            answer_type="probability",
            weights=REASONING_CONFIG["AGGREGATION_WEIGTHTS"],
            meta_model_name=REASONING_CONFIG["AGGREGATION_MODEL_NAME"],
            meta_prompt_template=REASONING_CONFIG["AGGREGATION_PROMPT_TEMPLATE"],
            meta_temperature=REASONING_CONFIG["AGGREGATION_TEMPERATURE"],
        )

        return ensemble_dict["meta_prediction"]

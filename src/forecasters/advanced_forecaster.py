from .forecaster import Forecaster
from common.datatypes import ForecastingQuestion

# llm_forecasting imports
from forecasters.llm_forecasting.prompts.prompts import PROMPT_DICT
from forecasters.llm_forecasting.utils.time_utils import (
    get_todays_date,
    subtract_days_from_date,
)
import forecasters.llm_forecasting.ranking as ranking
import forecasters.llm_forecasting.summarize as summarize
import forecasters.llm_forecasting.ensemble as ensemble
from forecasters.llm_forecasting.config.constants import DEFAULT_RETRIEVAL_CONFIG

import asyncio
from dataclasses import dataclass
from common.datatypes import DictLikeDataclass


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
    "AGGREGATION_METHOD": "meta",
    "AGGREGATION_PROMPT_TEMPLATE": PROMPT_DICT["meta_reasoning"]["0"],
    "AGGREGATION_TEMPERATURE": 0.2,
    "AGGREGATION_MODEL_NAME": "gpt-4",
    "AGGREGATION_WEIGHTS": None,
}


@dataclass
class RetrievalConfig(DictLikeDataclass):
    NUM_SEARCH_QUERY_KEYWORDS: int
    MAX_WORDS_NEWSCATCHER: int
    MAX_WORDS_GNEWS: int
    SEARCH_QUERY_MODEL_NAME: str
    SEARCH_QUERY_TEMPERATURE: float
    SEARCH_QUERY_PROMPT_TEMPLATES: list[str]
    NUM_ARTICLES_PER_QUERY: int
    SUMMARIZATION_MODEL_NAME: str
    SUMMARIZATION_TEMPERATURE: float
    SUMMARIZATION_PROMPT_TEMPLATE: str
    PRE_FILTER_WITH_EMBEDDING: bool
    PRE_FILTER_WITH_EMBEDDING_THRESHOLD: float
    RANKING_MODEL_NAME: str
    RANKING_TEMPERATURE: float
    RANKING_PROMPT_TEMPLATE: str
    RANKING_RELEVANCE_THRESHOLD: int
    RANKING_COSINE_SIMILARITY_THRESHOLD: float
    SORT_BY: str
    RANKING_METHOD: str
    RANKING_METHOD_LLM: str
    NUM_SUMMARIES_THRESHOLD: int
    EXTRACT_BACKGROUND_URLS: bool


@dataclass
class ReasoningConfig(DictLikeDataclass):
    BASE_REASONING_MODEL_NAMES: list[str]
    BASE_REASONING_TEMPERATURE: float
    BASE_REASONING_PROMPT_TEMPLATES: list[list[str]]
    AGGREGATION_METHOD: str
    AGGREGATION_PROMPT_TEMPLATE: str
    AGGREGATION_TEMPERATURE: float
    AGGREGATION_MODEL_NAME: str
    AGGREGATION_WEIGHTS: list[float]


RETRIEVAL_CONFIG: RetrievalConfig = RetrievalConfig(**DEFAULT_RETRIEVAL_CONFIG)
REASONING_CONFIG: ReasoningConfig = ReasoningConfig(**REASONING_CONFIG)


class AdvancedForecaster(Forecaster):
    def __init__(
        self,
        retrieval_config: RetrievalConfig | None = None,
        reasoning_config: ReasoningConfig | None = None,
        **kwargs,
    ):
        """
        Only override kwargs for retrieval_config if the retrieval_config is None.
        Only override kwargs for reasoning_config if the reasoning_config is None.
        """
        print("Loading AdvancedForecaster...")
        self.retrieval_config = retrieval_config or RETRIEVAL_CONFIG
        self.reasoning_config = reasoning_config or REASONING_CONFIG
        if retrieval_config is None:
            for key, value in kwargs.items():
                if key in self.retrieval_config.keys():
                    print(f"Overriding retrieval_config: {key}={value}")
                    self.retrieval_config[key] = value
        if reasoning_config is None:
            for key, value in kwargs.items():
                if key in self.reasoning_config.keys():
                    print(f"Overriding reasoning_config: {key}:={value}")
                    self.reasoning_config[key] = value
        if retrieval_config is not None and reasoning_config is not None and kwargs:
            print(
                "WARNING: kwargs passed to AdvancedForecaster constructor are not used."
            )
            print(f"kwargs: {kwargs}")

        print("Initialized forecaster with settings:")
        # print(f"Retrieval config: {self.retrieval_config}")
        # print(f"Reasoning config: {self.reasoning_config}")

    async def call_async(self, sentence: ForecastingQuestion, **kwargs) -> float:
        question = sentence.title
        background_info = (
            sentence.metadata["background_info"]
            if getattr(sentence, "metadata", None)
            and "background_info" in sentence.metadata
            else ""
        )
        resolution_criteria = (
            sentence.body
        )  # resolution criteria and other info is in |body|

        today_date = get_todays_date()
        # If open date is set in data structure, change beginning of retrieval to question open date.
        # Retrieve from [today's date - 1 month, today's date].
        retrieval_dates = (
            subtract_days_from_date(today_date, 30),
            today_date,
        )

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
            config=self.retrieval_config,
            return_intermediates=True,
        )

        all_summaries = summarize.concat_summaries(
            ranked_articles[: self.retrieval_config["NUM_SUMMARIES_THRESHOLD"]]
        )

        close_date = "N/A"  # data doesn't have explicit close date, so set to N/A
        today_to_close_date = [today_date, close_date]
        ensemble_dict = await ensemble.meta_reason(
            question=question,
            background_info=background_info,
            resolution_criteria=resolution_criteria,
            today_to_close_date_range=today_to_close_date,
            retrieved_info=all_summaries,
            reasoning_prompt_templates=self.reasoning_config[
                "BASE_REASONING_PROMPT_TEMPLATES"
            ],
            base_model_names=self.reasoning_config["BASE_REASONING_MODEL_NAMES"],
            base_temperature=self.reasoning_config["BASE_REASONING_TEMPERATURE"],
            aggregation_method=self.reasoning_config["AGGREGATION_METHOD"],
            weights=self.reasoning_config["AGGREGATION_WEIGHTS"],
            meta_model_name=self.reasoning_config["AGGREGATION_MODEL_NAME"],
            meta_prompt_template=self.reasoning_config["AGGREGATION_PROMPT_TEMPLATE"],
            meta_temperature=self.reasoning_config["AGGREGATION_TEMPERATURE"],
        )

        return ensemble_dict["meta_prediction"]

    def call(self, sentence: ForecastingQuestion, **kwargs) -> float:
        # This won't work inside a Jupyter notebook or similar; but there you can use await
        return asyncio.run(self.call_async(sentence, **kwargs))

    def dump_config(self):
        return {
            "retrieval_config": self.retrieval_config.to_dict(),
            "reasoning_config": self.reasoning_config.to_dict(),
        }


# TODO: make a cheaper/faster version of this that uses a different default config

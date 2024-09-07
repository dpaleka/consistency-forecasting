from common.perscache import register_model_for_cache
from pydantic import BaseModel
from typing import Optional


class ForecastingQuestion_stripped_with_resolution(BaseModel):
    title: str
    body: str
    resolution: bool


register_model_for_cache(ForecastingQuestion_stripped_with_resolution)


class ForecastingQuestion_stripped_with_resolution_list(BaseModel):
    questions: list[ForecastingQuestion_stripped_with_resolution]


register_model_for_cache(ForecastingQuestion_stripped_with_resolution_list)


class ForecastingQuestionGroundTruthResolutionStrict(BaseModel):
    resolution: Optional[bool]
    reasoning: str


register_model_for_cache(ForecastingQuestionGroundTruthResolutionStrict)


class ForecastingQuestionGroundTruthResolutionLax(BaseModel):
    resolution: bool
    reasoning: str


register_model_for_cache(ForecastingQuestionGroundTruthResolutionLax)

from common.perscache import register_model_for_cache
from pydantic import BaseModel


class ForecastingQuestion_stripped_with_resolution(BaseModel):
    title: str
    body: str
    resolution: bool


register_model_for_cache(ForecastingQuestion_stripped_with_resolution)


class ForecastingQuestion_stripped_with_resolution_list(BaseModel):
    questions: list[ForecastingQuestion_stripped_with_resolution]


register_model_for_cache(ForecastingQuestion_stripped_with_resolution_list)


class ForecastingQuestionGroundTruthResolution(BaseModel):
    resolution: bool

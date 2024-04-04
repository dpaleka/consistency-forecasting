from typing import Optional
from datetime import datetime
from uuid import uuid4, UUID
from pydantic import BaseModel, Field, field_validator


class PlainText(BaseModel):
    text: str


class Prob(BaseModel):
    prob: float

    @field_validator("prob")
    @classmethod
    def validate_prob(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Probability must be between 0 and 1.")
        return v


class Prob_cot(Prob):
    chain_of_thought: str
    prob: float  # redefine to maintain order


# this is what we pass to llms for instantiation and forecasting
# and also the response_model we expect from llms
class ForecastingQuestion_simple(BaseModel):
    title: str
    body: str

    def cast_FQ(
        self,
        resolution_date: datetime,
        question_type: str,
        data_source: Optional[str] = None,
        **kwargs
    ):
        """Make ForecastingQuestion from a ForecastingQuestion_simple given to us by an llm

        Args:
            resolution_date (datetime): If produced by an LLM, will usually be the max of the 
                resolution dates of the questions in the input.
            question_type (str): If produced by an LLM, will usually be the same as that of 
                the inputs
            data_source (Optional[str], optional): If produced by an LLM, will usually be
                "synthetic_inst". Defaults to None.
        """
        return ForecastingQuestion(
            title=self.title,
            body=self.body,
            resolution_date=resolution_date,
            question_type=question_type,
            data_source=data_source,
            **kwargs
        )


class ForecastingQuestion(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    body: str
    resolution_date: datetime
    question_type: str
    data_source: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[dict] = None
    resolution: Optional[str] = None

    @field_validator("question_type")
    def validate_question_type(cls, v):
        if v not in ["binary", "conditional_binary"]:
            raise ValueError(
                "Question type must be either 'binary' or 'conditional_binary'"
            )
        return v

    @field_validator("data_source")
    def validate_data_source(cls, v):
        if v not in [
            None,
            "synthetic",
            "synthetic_inst",
            "manifold",
            "metaculus",
            "predictit",
        ]:
            raise ValueError(
                'Data source must be in [None, "synthetic", "synthetic_inst", "manifold", "metaculus", "predictit"]'
            )
        return v

    def expected_answer_type(self, mode="default") -> type:
        exp_answer_types = {
            "default": {"binary": Prob, "conditional_binary": Prob},
            "cot": {"binary": Prob_cot, "conditional_binary": Prob_cot},
        }
        return exp_answer_types[mode][self.question_type]

    def cast_simple(self):
        return ForecastingQuestion_simple(title=self.title, body=self.body)

    def __str__(self):
        return self.cast_simple().model_dump_json()




# ForecastingQuestionTuple = dict[str, ForecastingQuestion]
# ProbsTuple = dict[str, Prob]

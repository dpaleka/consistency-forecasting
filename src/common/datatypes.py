from datetime import datetime
from uuid import uuid4, UUID
from pydantic import BaseModel, Field, field_validator


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
    prob: float  # redefine prob to maintain order

# this is what we pass to llms for instantiation and forecasting
# and also the response_model we expect from llms
class ForecastingQuestion_simple(BaseModel):
    title : str
    body : str
    
    def complex(self, **kwargs):
        """Make ForecastingQuestion from a ForecastingQuestion_simple given to us by an llm
        Mandatory kwargs: resolution_date, question_type. Recommended: data_source.
        """
        return ForecastingQuestion(
            title = self.title,
            body = self.body,
            **kwargs
        )
    

class ForecastingQuestion(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    body: str
    resolution_date: datetime
    question_type: str
    data_source: str = None
    url: str = None
    metadata: dict = None
    resolution: str = None

    @field_validator("question_type")
    def validate_question_type(cls, v):
        if v not in ["binary", "conditional_binary"]:
            raise ValueError(
                "Question type must be either 'binary' or 'conditional_binary'"
            )
        return v

    def expected_answer_type(self, mode="default") -> BaseModel:
        exp_answer_types = {
            "default": {
                "binary": Prob, 
                "conditional_binary": Prob
                },
            "chain_of_thought": {
                "binary": Prob_cot,
                "conditional_binary": Prob_cot
            }
        }
        return exp_answer_types[mode][self.question_type]
    
    def simple(self):
        return ForecastingQuestion_simple(
            title = self.title,
            body = self.body
        )
    
    def __str__(self):
        return self.simple().model_dump_json()


ForecastingQuestionTuple = dict[str, ForecastingQuestion]
ProbsTuple = dict[str, Prob]

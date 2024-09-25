from typing import Optional, Type, List
from datetime import datetime
from uuid import uuid4, UUID
from pydantic import BaseModel, Field, validator, field_validator, create_model
from .perscache import register_model_for_cache, register_models_for_cache
from enum import Enum


### Pydantic models ###
class PlainText(BaseModel):
    text: str


register_model_for_cache(PlainText)


class Prob(BaseModel):
    prob: float

    @field_validator("prob")
    @classmethod
    def validate_prob(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Probability must be between 0 and 1.")
        return v


register_model_for_cache(Prob)


class Forecast(BaseModel):
    metadata: list | dict | None = None
    prob: float

    @field_validator("prob")
    @classmethod
    def validate_prob(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Probability must be between 0 and 1.")
        return v

    def to_dict(self):
        return self.dict()


register_model_for_cache(Forecast)


class Prob_cot(BaseModel):
    chain_of_thought: str
    prob: float

    @field_validator("prob")
    @classmethod
    def validate_prob(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Probability must be between 0 and 1.")
        return v


register_model_for_cache(Prob_cot)


def reasoning_field(response: BaseModel) -> str:
    if isinstance(response, Prob_cot):
        return response.chain_of_thought
    elif isinstance(response, PlainText):
        return response.text
    elif isinstance(response, Prob):
        return str(response.prob)
    else:
        raise ValueError(
            f"Unsupported response model for extracting reasoning field: {response}"
        )


# this is what we pass to llms for instantiation and forecasting
# and also the response_model we expect from llms
class ForecastingQuestion_stripped(BaseModel):
    title: str
    body: str

    def cast_FQ(
        self,
        resolution_date: datetime,
        question_type: str,
        created_date: datetime = None,
        data_source: Optional[str] = None,
        **kwargs,
    ):
        """Make ForecastingQuestion from a ForecastingQuestion_stripped given to us by an llm

        Args:
            resolution_date (datetime): If produced by an LLM, will usually be the max of the
                resolution dates of the questions in the input.
            question_type (str): If produced by an LLM, will usually be the same as that of
                the inputs
            data_source (Optional[str], optional): If produced by an LLM, will usually be
                "synthetic_inst". Defaults to None.

        Keyword Args:
            url (Optional[str]): You probably shouldn't add this.
            metadata (Optional[dict]): Metadata.
            resolution (Optional[bool]): The resolution of the question.
        """
        return ForecastingQuestion(
            title=self.title,
            body=self.body,
            resolution_date=resolution_date,
            question_type=question_type,
            data_source=data_source,
            created_date=created_date,
            **kwargs,
        )

    def cast_stripped(self):
        return self


register_model_for_cache(ForecastingQuestion_stripped)


class ForecastingQuestion_stripped_list(BaseModel):
    questions: list[ForecastingQuestion_stripped]


register_model_for_cache(ForecastingQuestion_stripped_list)

exp_answer_types = {
    "default": {"binary": Prob, "conditional_binary": Prob},
    "cot": {"binary": Prob_cot, "conditional_binary": Prob_cot},
}


class ForecastingQuestion(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    body: str
    resolution_date: datetime
    question_type: str
    data_source: Optional[str] = None
    created_date: Optional[datetime] = None
    url: Optional[str] = None
    metadata: Optional[dict] = None
    resolution: Optional[bool] = None

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
        return exp_answer_types[mode][self.question_type]

    def cast_stripped(self) -> ForecastingQuestion_stripped:
        return ForecastingQuestion_stripped(title=self.title, body=self.body)

    def cast_FQ(self):
        return self

    def __str__(self) -> str:
        return self.cast_stripped().model_dump_json()

    def to_dict_forecast_mode(self, mode="default") -> dict:
        return_dict = {
            "title": self.title,
            "body": self.body,
            "resolution_date": str(self.resolution_date),
        }
        if self.created_date:
            return_dict["created_date"] = str(self.created_date)

        return return_dict

    def to_str_forecast_mode(self, mode="default") -> str:
        return str(self.to_dict_forecast_mode(mode=mode))

    def to_dict(self):
        return self.dict()


register_model_for_cache(ForecastingQuestion)


class ForecastingQuestions(BaseModel):
    questions: list[ForecastingQuestion]


register_model_for_cache(ForecastingQuestions)

# e.g. fields = = {'P' : 'binary', 'Q' : 'numerical', 'not_P' : 'binary'}


def mk_TupleFormat(fields: dict[str, str], name="TupleFormat") -> Type[BaseModel]:
    model = create_model(name, **{k: (ForecastingQuestion, ...) for k in fields})
    for field_name, field_type in fields.items():

        def make_validator(field_name: str, field_type: str):
            @validator(field_name, allow_reuse=True)
            def validate(cls, v):
                if v.question_type != field_type:
                    raise ValueError(f"{field_name}.question_type must be {field_type}")
                return v

            return validate

        model.add_validator(make_validator(field_name, field_type))
    return model


def mk_TupleFormat_ans(
    fields: dict[str, str], name="TupleFormat_ans"
) -> Type[BaseModel]:
    return create_model(
        name,
        **{
            field_name: (exp_answer_types["default"][field_type], ...)
            for field_name, field_type in fields.items()
        },
    )


ForecastingQuestionTuple = dict[str, ForecastingQuestion]
ProbsTuple = dict[str, Prob]


class ValidationResult(BaseModel):
    reasoning: str
    valid: bool


register_model_for_cache(ValidationResult)


class VerificationResult(BaseModel):
    reasoning: str
    valid: bool


register_model_for_cache(VerificationResult)


class RelevanceResult(BaseModel):
    reasons: list[str]
    conclusion: str
    score: float


register_model_for_cache(RelevanceResult)


class BodyAndDate(BaseModel):
    resolution_date: datetime
    resolution_criteria: str


register_model_for_cache(BodyAndDate)


class ResolutionDate(BaseModel):
    resolution_date: datetime


register_model_for_cache(ResolutionDate)


class SyntheticTagQuestion(BaseModel):
    title: str
    category: str
    tags: str
    feedback: Optional[str] = None
    fixed: Optional[bool] = False
    body: Optional[str] = None
    resolution_date: Optional[str] = None


register_model_for_cache(SyntheticTagQuestion)


class SyntheticRelQuestion(BaseModel):
    title: str
    body: Optional[str] = None
    resolution_date: Optional[str] = None
    source_question: Optional[str] = None
    feedback: Optional[str] = None
    fixed: Optional[bool] = False


register_model_for_cache(SyntheticRelQuestion)


class QuestionGenerationResponse(BaseModel):
    questions: list[SyntheticRelQuestion]


class QuestionGenerationResponse_FQ(BaseModel):
    questions: list[ForecastingQuestion_stripped]


register_model_for_cache(QuestionGenerationResponse)
register_model_for_cache(QuestionGenerationResponse_FQ)


class QuestionGenerationResponse3(BaseModel):
    question_1: SyntheticTagQuestion
    question_2: SyntheticTagQuestion
    question_3: SyntheticTagQuestion


register_model_for_cache(QuestionGenerationResponse3)


class ResolverOutput(BaseModel):
    chain_of_thought: str
    can_resolve_question: bool
    answer: Optional[bool]


register_model_for_cache(ResolverOutput)

### end Pydantic models ###

### Other useful classes


class DictLikeDataclass:
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def keys(self):
        return self.__annotations__.keys()

    def values(self):
        return (getattr(self, key) for key in self.keys())

    def items(self):
        return ((key, getattr(self, key)) for key in self.keys())

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.keys()}


class Consequence_ConsequenceType(str, Enum):
    quantity = "quantity"
    time = "time"
    misc = "misc"
    none = "none"


class Consequence_ClassifyOutput(BaseModel):
    consequence_type: List[Consequence_ConsequenceType]


class Consequence_InstantiateOutput(BaseModel):
    title: str
    body: str
    resolution_date: datetime


register_models_for_cache([Consequence_ClassifyOutput, Consequence_InstantiateOutput])

from typing import Optional, Type
from datetime import datetime
from uuid import uuid4, UUID
from pydantic import BaseModel, Field, validator, field_validator, create_model


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
class ForecastingQuestion_stripped(BaseModel):
    title: str
    body: str

    def cast_FQ(
        self,
        resolution_date: datetime,
        question_type: str,
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
            resolution (Optional[str]): The resolution of the question.
        """
        return ForecastingQuestion(
            title=self.title,
            body=self.body,
            resolution_date=resolution_date,
            question_type=question_type,
            data_source=data_source,
            **kwargs,
        )
    
    def cast_stripped(self):
        return self


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

    def cast_stripped(self):
        return ForecastingQuestion_stripped(title=self.title, body=self.body)
    
    def cast_FQ(self):
        return self

    def __str__(self):
        return self.cast_stripped().model_dump_json(indent=4)


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

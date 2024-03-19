from datetime import datetime
from dateutil import parser
import re
from typing import Dict

class Prob(float):
    def __new__(cls, value):
        if not (0.0 <= value <= 1.0):
            raise ValueError("Probability must be between 0 and 1.")
        return super(Prob, cls).__new__(cls, value)


class QuestionType(str):
    def __new__(cls, value):
        if value not in ["binary", "conditional_binary"]:
            raise ValueError(
                "Question type must be one of 'binary', 'conditional_binary'."
            )
        return super(QuestionType, cls).__new__(cls, value)

    exp_answers = {"binary": Prob, "conditional_binary": Prob}

    def expected_answer_type(self) -> str:
        return self.exp_answers.get(self, None)


class Sentence:
    def __init__(
        self,
        title: str,  # aka "text"
        body: str,  # aka "resolution_criteria"
        resolution_date: datetime | None,
        question_type: QuestionType,
        data_source: str | None = None,  # e.g. synthetic, metaculus, manifold, predictit
        url: str | None  = None,
        metadata: dict = None,  # for example, topics : list[str]
        resolution: str | None = None,  # some questions may already have been resolved
    ):
        # self.id = TODO
        self.title = title
        self.body = body
        self.resolution_date = resolution_date
        self.question_type = question_type
        self.data_source = data_source
        self.url = url
        self.metadata = metadata
        self.resolution = resolution

    def to_dict(self) -> dict:
        return {
            # "id": self.id, # TODO
            "title": self.title,
            "body": self.body,
            "resolution_date": (
                self.resolution_date.isoformat() if self.resolution_date else None
            ),
            "question_type": self.question_type,
            "data_source": self.data_source,
            "url": self.url,
            "metadata": self.metadata,
            "resolution": self.resolution,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Sentence":
        return cls(
            # id=d["id"],
            title=d["title"],
            body=d["body"],
            resolution_date=(
                datetime.fromisoformat(d["resolution_date"])
                if d["resolution_date"]
                else None
            ),
            question_type=QuestionType(d["question_type"]),
            data_source=d["data_source"],
            url=d["url"],
            metadata=d["metadata"],
            resolution=d["resolution"],
        )

    def __str__(self):
        return (
            f"TITLE: {self.title}\n"
            f"RESOLUTION DATE: {self.resolution_date}\n"
            f"DETAILS:\n{self.body}"
        )

    @classmethod
    def from_str(
        cls,
        string: str,
        question_type: QuestionType,
        **kwargs,  # url, data_source, metadata, resolution
    ) -> "Sentence":
        title = re.search(r"TITLE:(.*?)", string).group(1)
        resolution_date = re.search(r"RESOLUTION DATE: (.*?)\n", string).group(1)
        body = re.search(r"DETAILS:(.*?)$", string, re.DOTALL).group(1)
        return cls(
            title=title,
            body=body,
            resolution_date=parser.parse(resolution_date),
            question_type=question_type,
            **kwargs,
        )


SentencesTemplate = Dict[str, Sentence]
ProbsTemplate = Dict[str, Prob]
#%%
import json
from common.datatypes import ForecastingQuestion_stripped, ForecastingQuestion, Prob_cot, Prob

fq = ForecastingQuestion(
    title="Will Manhattan have a skyscraper a mile tall by 2030?",
    body=(
        "Resolves YES if at any point before 2030, there is at least "
        "one building in the NYC Borough of Manhattan (based on current "
        "geographic boundaries) that is at least a mile tall."
    ),
    resolution_date="2030-01-01T00:00:00",
    question_type="binary",
    data_source="manifold",
    url="https://www.metaculus.com/questions/12345/",
    metadata={"foo": "bar"},
    resolution=None,
)

fqs = ForecastingQuestion_stripped(
    title="Will Manhattan have a skyscraper a mile tall by 2030?",
    body=(
        "Resolves YES if at any point before 2030, there is at least "
        "one building in the NYC Borough of Manhattan (based on current "
        "geographic boundaries) that is at least a mile tall."
    ),
)


print(fqs.__str__())


# %%

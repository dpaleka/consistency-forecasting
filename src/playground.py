# %%

from static_checks.MiniInstantiator import RelevantInfo, ForecastingQuestion, datetime
from static_checks.Checker import BidChecker

# ri = RelevantInfo()
bc = BidChecker()

base_sentences = [
    ForecastingQuestion(
        title="Will the Labour Party win the next UK general election?",
        body="Resolves YES if the Labour Party wins the next UK general election.",
        resolution_date=datetime(2024, 5, 1),
        question_type="binary",
        data_source="synthetic",
    ),
]

# x=ri.instantiate_sync(base_sentences, relevance="mid")

# x=await bc.instantiate(base_sentences)

# %%

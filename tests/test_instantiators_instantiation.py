"""TEST INSTANTIATORS"""

# from common.utils import *
# from common.llm_utils import *
from datetime import datetime

# from static_checks.MiniInstantiator import *
from static_checks import Checker
from static_checks.Checker import (
    NegChecker,
    AndChecker,
    OrChecker,
    AndOrChecker,
    ButChecker,
    CondChecker,
    ConsequenceChecker,
    ParaphraseChecker,
    SymmetryAndChecker,
    SymmetryOrChecker,
    CondCondChecker,
)
from common.datatypes import ForecastingQuestion
from common.path_utils import get_data_path
from instantiation import instantiate
from pathlib import Path
import pytest

BASE_DATA_PATH: Path = (
    get_data_path() / "fq" / "real" / "questions_cleaned_formatted.jsonl"
)
TUPLES_PATH: Path = get_data_path() / "test" / "instantiation/"

checkers: dict[str, Checker] = {
    "NegChecker": NegChecker(path=TUPLES_PATH / "NegChecker.jsonl"),
    "AndChecker": AndChecker(path=TUPLES_PATH / "AndChecker.jsonl"),
    "OrChecker": OrChecker(path=TUPLES_PATH / "OrChecker.jsonl"),
    "AndOrChecker": AndOrChecker(path=TUPLES_PATH / "AndOrChecker.jsonl"),
    "ButChecker": ButChecker(path=TUPLES_PATH / "ButChecker.jsonl"),
    "CondChecker": CondChecker(path=TUPLES_PATH / "CondChecker.jsonl"),
    "ConsequenceChecker": ConsequenceChecker(
        path=TUPLES_PATH / "ConsequenceChecker.jsonl"
    ),
    "ParaphraseChecker": ParaphraseChecker(
        path=TUPLES_PATH / "ParaphraseChecker.jsonl"
    ),
    "SymmetryAndChecker": SymmetryAndChecker(
        path=TUPLES_PATH / "SymmetryAndChecker.jsonl"
    ),
    "SymmetryOrChecker": SymmetryOrChecker(
        path=TUPLES_PATH / "SymmetryOrChecker.jsonl"
    ),
    "CondCondChecker": CondCondChecker(path=TUPLES_PATH / "CondCondChecker.jsonl"),
}


@pytest.mark.asyncio
async def test_instantiate():
    await instantiate(
        BASE_DATA_PATH=BASE_DATA_PATH, checker_list=checkers, n_relevance=2, length=1
    )

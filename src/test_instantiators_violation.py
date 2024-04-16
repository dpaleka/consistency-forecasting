from forecasters import BasicForecaster
from static_checks.Checker import NegChecker, AndChecker, OrChecker, AndOrChecker, ButChecker, CondChecker, ConsequenceChecker, ParaphraseChecker, SymmetryAndChecker, SymmetryOrChecker, CondCondChecker

basic_forecaster = BasicForecaster()

BASE_DATA_PATH = "src/data/"

# model = gpt-3.5-turbo
model = "gpt-4-turbo-2024-04-09"

neg_checker = NegChecker(path=BASE_DATA_PATH + "NegChecker.jsonl")
and_checker = AndChecker(path=BASE_DATA_PATH + "AndChecker.jsonl")
or_checker = OrChecker(path=BASE_DATA_PATH + "OrChecker.jsonl")
andor_checker = AndOrChecker(path=BASE_DATA_PATH + "AndOrChecker.jsonl")
but_checker = ButChecker(path=BASE_DATA_PATH + "ButChecker.jsonl")
cond_checker = CondChecker(path=BASE_DATA_PATH + "CondChecker.jsonl")
cons_checker = ConsequenceChecker(path=BASE_DATA_PATH + "ConsequenceChecker.jsonl")
para_checker = ParaphraseChecker(path=BASE_DATA_PATH + "ParaphraseChecker.jsonl")
symmand_checker = SymmetryAndChecker(path=BASE_DATA_PATH + "SymmetryAndChecker.jsonl")
symmor_checker = SymmetryOrChecker(path=BASE_DATA_PATH + "SymmetryOrChecker.jsonl")
condcond_checker = CondCondChecker(path=BASE_DATA_PATH + "CondCondChecker.jsonl")
# neg_checker.test(basic_forecaster, model = model)
# and_checker.test(basic_forecaster, model = model)
# or_checker.test(basic_forecaster, model = model)
# andor_checker.test(basic_forecaster, model = model)
# but_checker.test(basic_forecaster, model=model)
# cond_checker.test(basic_forecaster, model=model)
# cons_checker.test(basic_forecaster, model=model)
# para_checker.test(basic_forecaster, model=model)
# symmand_checker.test(basic_forecaster, model=model)
# symmor_checker.test(basic_forecaster, model=model)
condcond_checker.test(basic_forecaster, model=model)

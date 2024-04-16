from forecasters import BasicForecaster
from static_checks.Checker import NegChecker, AndChecker, OrChecker, AndOrChecker, ButChecker, CondChecker, ConsequenceChecker, ParaphraseChecker, SymmetryAndChecker, SymmetryOrChecker, CondCondChecker

basic_forecaster = BasicForecaster()

neg_checker = NegChecker(path="src/data/NegChecker.jsonl")
and_checker = AndChecker(path="src/data/AndChecker.jsonl")
or_checker = OrChecker(path="src/data/OrChecker.jsonl")
andor_checker = AndOrChecker(path="src/data/AndOrChecker.jsonl")
but_checker = ButChecker(path="src/data/ButChecker.jsonl")
cond_checker = CondChecker(path="src/data/CondChecker.jsonl")
cons_checker = ConsequenceChecker(path="src/data/ConsequenceChecker.jsonl")
para_checker = ParaphraseChecker(path="src/data/ParaphraseChecker.jsonl")
symmand_checker = SymmetryAndChecker(path="src/data/SymmetryAndChecker.jsonl")
symmor_checker = SymmetryOrChecker(path="src/data/SymmetryOrChecker.jsonl")
condcond_checker = CondCondChecker(path="src/data/CondCondChecker.jsonl")

# neg_checker.test(basic_forecaster, model = "gpt-3.5-turbo")
# and_checker.test(basic_forecaster, model = "gpt-3.5-turbo")
# or_checker.test(basic_forecaster, model = "gpt-3.5-turbo")
# andor_checker.test(basic_forecaster, model = "gpt-3.5-turbo")
# but_checker.test(basic_forecaster, model="gpt-3.5-turbo")
# cond_checker.test(basic_forecaster, model="gpt-3.5-turbo")
# cons_checker.test(basic_forecaster, model="gpt-3.5-turbo")
# para_checker.test(basic_forecaster, model="gpt-3.5-turbo")
# symmand_checker.test(basic_forecaster, model="gpt-3.5-turbo")
# symmor_checker.test(basic_forecaster, model="gpt-3.5-turbo")
condcond_checker.test(basic_forecaster, model="gpt-3.5-turbo")

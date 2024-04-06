from forecasters import BasicForecaster
from static_checks.MiniInstantiator import *
from static_checks.Checker import *

basic_forecaster = BasicForecaster()

neg_checker = NegChecker() # or NegationChecker(path = "src/data/NegationChecker.jsonl")
and_checker = AndChecker()
or_checker = OrChecker()
andor_checker = AndOrChecker()
but_checker = ButChecker()
cond_checker = CondChecker()
cons_checker = ConsequenceChecker()
para_checker = ParaphraseChecker()
symmand_checker = SymmetryAndChecker()
symmor_checker = SymmetryOrChecker()

# neg_checker.test(basic_forecaster, model = "gpt-3.5-turbo")
# and_checker.test(basic_forecaster, model = "gpt-3.5-turbo")
# or_checker.test(basic_forecaster, model = "gpt-3.5-turbo")
# andor_checker.test(basic_forecaster, model = "gpt-3.5-turbo")
# but_checker.test(basic_forecaster, model="gpt-3.5-turbo")
# cond_checker.test(basic_forecaster, model="gpt-3.5-turbo")
# cons_checker.test(basic_forecaster, model="gpt-3.5-turbo")
# para_checker.test(basic_forecaster, model="gpt-3.5-turbo")
# symmand_checker.test(basic_forecaster, model="gpt-3.5-turbo")
symmor_checker.test(basic_forecaster, model="gpt-3.5-turbo")

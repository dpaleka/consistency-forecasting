from forecasters import BasicForecaster
from static_checks.MiniInstantiator import *
from static_checks.Checker import *

basic_forecaster = BasicForecaster()

neg_checker = NegChecker() # or NegationChecker(path = "src/data/NegationChecker.jsonl")
and_checker = AndChecker()
or_checker = OrChecker()
andor_checker = AndOrChecker()

#neg_checker.test(basic_forecaster, model = "gpt-3.5-turbo")
and_checker.test(basic_forecaster, model = "gpt-3.5-turbo")
or_checker.test(basic_forecaster, model = "gpt-3.5-turbo")
andor_checker.test(basic_forecaster, model = "gpt-3.5-turbo")

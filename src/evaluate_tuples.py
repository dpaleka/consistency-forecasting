from forecasters import BasicForecaster
from static_checks import *

basic_forecaster = BasicForecaster()

negation_checker = NegChecker() # or NegationChecker(path = "src/data/NegationChecker.jsonl")

negation_checker.test(basic_forecaster, model = "gpt-3.5-turbo")
# butnot_checker.test(basic_forecaster)
# paraphrasal_checker.test(basic_forecaster)
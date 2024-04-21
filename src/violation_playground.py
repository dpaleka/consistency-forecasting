from forecasters import BasicForecaster
from static_checks.Checker import Checker, NegChecker#, AndChecker, OrChecker, AndOrChecker, ButChecker, CondChecker, ConsequenceChecker, ParaphraseChecker, SymmetryAndChecker, SymmetryOrChecker, CondCondChecker

#basic_forecaster = BasicForecaster()

neg_checker = NegChecker()

v=neg_checker.violation({
    "P": 0.5,
    "not_P": 0.3
})

print(v)
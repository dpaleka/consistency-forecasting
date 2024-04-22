from forecasters import BasicForecaster
from static_checks.Checker import Checker, NegChecker, AndChecker, OrChecker, AndOrChecker, ButChecker, CondChecker#, ConsequenceChecker, ParaphraseChecker, SymmetryAndChecker, SymmetryOrChecker, CondCondChecker

#basic_forecaster = BasicForecaster()

neg_checker = NegChecker()
and_checker = AndChecker()
or_checker = OrChecker()
andor_checker = AndOrChecker()
but_checker = ButChecker()
cond_checker = CondChecker()

v=cond_checker.max_min_arbitrage({
    "P": 0.5,
    "Q_given_P" : 0.3,
    "P_and_Q" : 0.35
})

print(v)
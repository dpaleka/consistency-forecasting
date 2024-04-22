from forecasters import BasicForecaster
from static_checks.Checker import (
    Checker,
    NegChecker,
    AndChecker,
    OrChecker,
    AndOrChecker,
    ButChecker,
    CondChecker,
    ConsequenceChecker,
)  # , ParaphraseChecker, SymmetryAndChecker, SymmetryOrChecker, CondCondChecker

# basic_forecaster = BasicForecaster()

neg_checker = NegChecker()
and_checker = AndChecker()
or_checker = OrChecker()
andor_checker = AndOrChecker()
but_checker = ButChecker()
cond_checker = CondChecker()
cons_checker = ConsequenceChecker()

# v = neg_checker.max_min_arbitrage(
#     {
#         "P": 0.5,
#         "not_P": 0.9,
#     }
# )

# v = and_checker.max_min_arbitrage(
#     {
#         "P": 0.5,
#         "Q": 0.3,
#         "P_and_Q": 0.9,
#     }
# )

# v = or_checker.max_min_arbitrage(
#     {
#         "P": 0.5,
#         "Q": 0.3,
#         "P_or_Q": 0.9,
#     }
# )

# v = andor_checker.max_min_arbitrage(
#     {
#         "P": 0.5,
#         "Q": 0.3,
#         "P_and_Q": 0.9,
#         "P_or_Q": 0.7,
#     }
# )

# v = but_checker.max_min_arbitrage(
#     {
#         "P": 0.5,
#         "P_or_Q": 0.3,
#         "Q_and_not_P": 0.9,
#     }
# )

# v = cons_checker.max_min_arbitrage(
#     {
#         "P": 0.5,
#         "cons_P": 0.7,
#     }
# )

# wrong
# v=cond_checker.max_min_arbitrage({
#     "P": 0.5,
#     "Q_given_P" : 0.3,
#     "P_and_Q" : 0.9
# })

print(v)

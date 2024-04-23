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

neg_checker = NegChecker()
and_checker = AndChecker()
or_checker = OrChecker()
and_or_checker = AndOrChecker()
but_checker = ButChecker()
cond_checker = CondChecker()
consequence_checker = ConsequenceChecker()

# Nelder-Mead, L-BFGS-B, trust-exact -- often unreliable, as they are local optimization
# basinhopping -- slow I think? at least for AndChecker, OrChecker, AndOrChecker
# brute -- some syntax error
# differential_evolution, shgo, dual_annealing -- working

# nc_arb = neg_checker.max_min_arbitrage(
#     answers={"P": 0.5, "not_P": 0.3},
#     initial_guess=[0.5, 0.3],
#     method="basinhopping",
# )
# ac_arb = and_checker.max_min_arbitrage(
#     answers={"P": 0.5, "Q": 0.3, "P_and_Q": 0.4},
#     initial_guess=[0.1, 0.4, 0.02],
#     method="basinhopping",
# )
# oc_arb = or_checker.max_min_arbitrage(
#     answers={"P": 0.5, "Q": 0.3, "P_or_Q": 0.4},
#     initial_guess=[0.1, 0.4, 0.02],
#     method="basinhopping",
# )
# aoc_arb = and_or_checker.max_min_arbitrage(
#     answers={"P": 0.5, "Q": 0.3, "P_and_Q": 0.4, "P_or_Q": 0.3},
#     initial_guess=[0.1, 0.4, 0.4, 0.1],
#     method="basinhopping",
# )
# bc_arb = but_checker.max_min_arbitrage(
#     answers={"Q_and_not_P": 0.5, "P": 0.3, "P_or_Q": 0.4},
#     initial_guess=[0.5,0.3,0.4],
#     method="shgo",
# )
# cd_arb = cond_checker.max_min_arbitrage(
#     answers={"P": 0.5, "Q_given_P": 0.8, "P_and_Q": 0.7},
#     initial_guess=[0.5, 0.8, 0.7],
#     method="shgo",
# )
# cs_arb = consequence_checker.max_min_arbitrage(
#     answers={"P": 0.5, "cons_P": 0.4},
#     initial_guess=[0.5, 0.5],
#     method="shgo",
# )


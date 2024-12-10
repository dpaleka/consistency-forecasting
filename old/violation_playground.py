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
    ParaphraseChecker,
    SymmetryAndChecker,
    SymmetryOrChecker,
    CondCondChecker,
)

neg_checker = NegChecker(path="")
and_checker = AndChecker(path="")
or_checker = OrChecker(path="")
and_or_checker = AndOrChecker(path="")
but_checker = ButChecker(path="")
cond_checker = CondChecker(path="")
consequence_checker = ConsequenceChecker(path="")
para_checker = ParaphraseChecker(path="")
sand_checker = SymmetryAndChecker(path="")
sor_checker = SymmetryOrChecker(path="")
cc_checker = CondCondChecker(path="")

# Nelder-Mead, L-BFGS-B, trust-exact -- often unreliable, as they are local optimization
# basinhopping -- slow I think? at least for AndChecker, OrChecker, AndOrChecker
# brute -- some syntax error
# differential_evolution, shgo, dual_annealing -- working

# nc_arb = neg_checker.max_min_arbitrage(
#     answers={"P": 0.5, "not_P": 0.3},
#     initial_guess=[0.5, 0.3],
#     method="shgo",
# )
# ac_arb = and_checker.max_min_arbitrage(
#     answers={"P": 0.5, "Q": 0.3, "P_and_Q": 0.4},
#     initial_guess=[0.1, 0.4, 0.02],
#     method="shgo",
# )
# oc_arb = or_checker.max_min_arbitrage(
#     answers={"P": 0.5, "Q": 0.3, "P_or_Q": 0.4},
#     initial_guess=[0.1, 0.4, 0.02],
#     method="shgo",
# )
# aoc_arb = and_or_checker.max_min_arbitrage(
#     answers={"P": 0.5, "Q": 0.3, "P_and_Q": 0.4, "P_or_Q": 0.3},
#     initial_guess=[0.1, 0.4, 0.4, 0.1],
#     method="shgo",
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
# p_arb = para_checker.max_min_arbitrage(
#     answers={"P": 0.5, "para_P": 0.4},
#     initial_guess=[0.5, 0.4],
#     method="shgo",
# )
# sand_arb = sand_checker.max_min_arbitrage(
#     answers={"P_and_Q": 0.5, "Q_and_P": 0.4},
#     initial_guess=[0.1, 0.4],
#     method="shgo",
# )
# sor_arb = sor_checker.max_min_arbitrage(
#     answers={"P_or_Q": 0.5, "Q_or_P": 0.4},
#     initial_guess=[0.1, 0.4],
#     method="shgo",
# )
# cc_arb = cc_checker.max_min_arbitrage(
#     answers={"P": 0.5, "Q_given_P": 0.4, "R_given_P_and_Q": 0.3, "P_and_Q_and_R": 0.2},
#     initial_guess=[0.5, 0.4, 0.3, 0.2],
#     method="shgo",
# )


# print(cc_arb)

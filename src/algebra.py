from sympy import *

# cond checker

priceP = symbols("F(P)")
priceQgivenP = symbols("F(Q|P)")
pricePandQ = symbols("F(P∧Q)")
consp = symbols("p")
consq = symbols("q")

profit1 = (
    lambda p, q: log(p)
    - log(priceP)
    + log(q)
    - log(priceQgivenP)
    + log(p * q)
    - log(pricePandQ)
)
profit2 = (
    lambda p, q: log(p)
    - log(priceP)
    + log(1 - q)
    - log(1 - priceQgivenP)
    + log(1 - p * q)
    - log(1 - pricePandQ)
)
profit3 = (
    lambda p, q: log(1 - p) - log(1 - priceP) + log(1 - p * q) - log(1 - pricePandQ)
)

# eq1 = Eq(
#     profit1(consp, consq),
#     profit2(consp, consq),
# )
# eq2 = Eq(
#     profit2(consp, consq),
#     profit3(consp, consq),
# )

# numsols = len(list(nonlinsolve([eq1, eq2], [consp, consq])))
# consp_, consq_ = list(nonlinsolve([eq1, eq2], [consp, consq]))[0]
# consp_ = consp_.args[0].args[0]
# consq_ = consq_.args[0].args[0]


# violation = profit1(consp_, consq_)

# print(numsols)
# print('-----------------')
# print(simplify(consp_))
# print('-----------------')
# print(simplify(consq_))
# print('-----------------')
# print(simplify(violation))
# print('-----------------')
# print(simplify(violation.subs(pricePandQ, priceP*priceQgivenP).subs(priceP, 0.3).subs(priceQgivenP, 0.5)))

# trying semi-automated approach

# exp1 = (consp * consq) ** 2 / (priceP * priceQgivenP * pricePandQ)
# exp2 = consp * (1 - consq) * (1 - consp * consq) / (priceP * (1 - priceQgivenP) * (1 - pricePandQ))
# exp3 = (1 - consp) * (1 - consp * consq) / ((1 - priceP) * (1 - pricePandQ))

# eq1 = Eq(
#     exp1,
#     exp2,
# )
# eq2 = Eq(
#     exp2,
#     exp3,
# )

# sols = nonlinsolve([eq1, eq2], [consp, consq])
# sols = list(sols)
# numsols = len(sols)
# print(simplify(sols[0]))

# even more semi-automated

# X = symbols("X")
# Y = symbols("Y")
# p = symbols("p")

# eq = Eq(
#     - (1 + Y * (1 - 1 / p)) / (Y * (1 + Y) * (1 - 1 / p) ** 2),
#     X
# )

# sol = solve(eq, p)

# print(sol)

# let's just see if the numbers are right at least

import math

p = 0.5
q = 0.3
pq = p * q

exp_result = (p * q) ** 2

# x = (
#     2 * (2 * p ** 2 * q ** 2 / ((1 - p)*(1 - pq)))
#     - p*(1 - q) / (1 - p)
#     - math.sqrt((p * (1 - q) / (1 - p)) ** 2 - 4*(pq / (1 - p)) ** 2)
# ) / (2 + 2 * (p ** 2 * q ** 2 / ((1 - p)*(1 - pq))))

X = p*q**2/((1-q)*(1-pq))
Y=p*(1-q)/(1-p)
D = Y**2-4*X*Y*(1+Y)
pq_ = (2*X*Y-Y-math.sqrt(D))/(2+2*X*Y)

print(exp_result)
print(pq_**2)

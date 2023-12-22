import numpy

from black_belt.ne import lp_solve_zero_sum, best_response

payoff_matrix = numpy.array([
    [ 0.5, 0.9, 0.1],
    [ 0.1, 0.5, 0.9],
    [ 0.9, 0.1, 0.5],
])

policy, value = lp_solve_zero_sum(payoff_matrix)
print(policy)
print(value)

policy, value = best_response(payoff_matrix, [0.3333, 0.3333, 0.3333])
print(policy)
print(value)

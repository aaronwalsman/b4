import numpy

from scipy.optimize import linprog

# Notes:
# So far, this seems to give the best result for the column player who is
# trying to maximize the game value.

payoff_matrix = numpy.array([
    [ 0.5,   1,   0],
    [   0, 0.5,   1],
    [   1,   1,   1],
])

b_ub = numpy.array([-1,-1,-1])

u = linprog(numpy.array([1,1,1]), A_ub = -payoff_matrix, b_ub = b_ub)

game_value = 1./numpy.sum(u.x)
policy = u.x * game_value
print(game_value)
print(policy)

def ne(game):
    
    pv = linprog(
        numpy.ones(game.shape[1]),
        A_ub=-game,
        b_ub=-numpy.ones(game.shape[0]),
    )
    
    value = 1. / numpy.sum(pv.x)
    policy = pv.x * value
    
    return policy, value

p, v = ne(payoff_matrix)
print(p, v)

pd, vd = ne(1.-payoff_matrix.T)
print(pd, vd)

example2 = numpy.array([
    [0,   4/6, 1/6],
    [4/6, 1/6, 5/6],
])

p, v = ne(example2)
print(p, v)

pd, vd = ne(1.-example2.T)
print(pd, vd)

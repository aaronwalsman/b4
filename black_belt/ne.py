import numpy

from scipy.optimize import linprog

def linear_zero_sum(game):
    opt = linprog(
        numpy.ones(game.shape[1]),
        A_ub=-game,
        b_ub=-numpy.ones(game.shape[0]),
        method='simplex',
        options={'tol':1e-6},
    )
    
    value = 1./numpy.sum(opt.x)
    policy = opt.x * value
    
    return policy, value

def best_response(game, opponent_distribution):
    r,c = game.shape
    a = numpy.array(opponent_distribution) @ game
    i = numpy.argmax(a)
    policy = numpy.zeros(c)
    policy[i] = 1.
    value = a[i]
    
    return policy, value

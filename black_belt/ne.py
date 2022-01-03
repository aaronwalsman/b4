import numpy

from scipy.optimize import linprog

def ne(game):
    opt = linprog(
        numpy.ones(game.shape[1]),
        A_ub=-game,
        b_ub=-numpy.ones(game.shape[0]),
        method='simplex',
    )
    
    value = 1./numpy.sum(opt.x)
    policy = opt.x * value
    
    return policy, value

import numpy as np

#from black_belt.ne import equilibrium
from black_belt.ne import lp_solve_zero_sum
import json

#game = numpy.load('dump_802416.npy')
#policy, value = equilibrium(game)

data = json.load(open('fail_log_18.json'))
game = np.array(data['game'])

print(game)

policy, value = lp_solve_zero_sum(game)

print(policy, value)

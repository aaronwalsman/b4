import numpy

from black_belt.ne import equilibrium

game = numpy.load('dump_802416.npy')

policy, value = equilibrium(game)

print(policy, value)

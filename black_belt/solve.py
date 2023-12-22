import time
import os
import sys
import multiprocessing
import pickle
import json
from zipfile import ZipFile
from argparse import ArgumentParser
import warnings

import numpy

from scipy.optimize import OptimizeWarning

import tqdm

from black_belt.game import game_mode, State
from black_belt.ne import lp_solve_zero_sum
from black_belt.game_statistics import (
    total_states,
    state_to_index,
    index_to_state,
    payoff_matrix,
)

'''
Solves the Bodega Brawl game by computing the Nash Equilibrium for every
possible game state.  This is accomplished by starting from the end of the
game and working backwards.  This is designed to be run on many processes
in parallel, and takes approximatesly 30 minutes to solve the large version
of the game with 40 parallel processes.
'''

# setup argument parser
parser = ArgumentParser()
parser.add_argument('--num-procs', type=int, default=40)

if __name__ == '__main__':
    
    # parse the arguments
    args = parser.parse_args()

    # setup multiprocessing and shared data
    context = multiprocessing.get_context(None)
    policy_shape = (9, total_states)
    policy = context.RawArray('d', policy_shape[0]*policy_shape[1])
    value = context.RawArray('d', total_states)
    complete = context.RawArray('d', total_states)
    status = context.RawArray('d', args.num_procs)

    # zero the shared data
    complete_np = numpy.frombuffer(complete, dtype=numpy.float64)
    numpy.copyto(complete_np, numpy.zeros(total_states))
    status_np = numpy.frombuffer(status, dtype=numpy.float64)
    numpy.copyto(status_np, numpy.zeros(args.num_procs))
    
    # worker function that will be launched in each new process
    def worker(proc_id):
        try:
            for i in range(proc_id, total_states, args.num_procs):
                if complete[i]:
                    continue
                game, p1_actions = payoff_matrix(
                    index_to_state(i), value, complete=complete)
                with warnings.catch_warnings():
                    warnings.simplefilter('error', OptimizeWarning)
                    try:
                        p, v = lp_solve_zero_sum(game)
                        state = index_to_state(i)
                    except:
                        failure_log = {
                            'index' : i,
                            'state' : index_to_state(i),
                            'game' : game.tolist(),
                            'error' : 'solve_error',
                        }
                        with open('./fail_log_%i.json'%proc_id, 'w') as f:
                            json.dump(failure_log, f, indent=2)
                            raise
                
                if numpy.any(numpy.isnan(v)) or numpy.any(numpy.isnan(p)):
                    failure_log = {
                        'index' : i,
                        'state' : index_to_state(i),
                        'game' : game.tolist(),
                        'v' : v,
                        'p' : p,
                        'error' : 'value_error',
                    }
                    with open('./fail_log_%i.json'%proc_id, 'w') as f:
                        json.dump(failure_log, f, indent=2)
                        raise Exception('nan value')
                        
                value[i] = v
                full_p = numpy.zeros(9)
                action_indices = [int(a) for a in p1_actions]
                full_p[action_indices] = p
                policy[i*9:(i+1)*9] = full_p
                complete[i] = 1
        except:
            status[proc_id] = 1
            raise
    
    # make new processes
    for i in range(args.num_procs):
        process = context.Process(
            target=worker,
            name='worker_%i'%i,
            args=(i,),
        )
        process.daemon = True
        process.start()
    
    # keep track of progress
    progress = tqdm.tqdm(total=total_states)
    last_complete = 0
    total_complete = 0
    while total_complete < total_states:
        total_complete = complete_np.sum()
        progress.update(int(total_complete - last_complete))
        last_complete = total_complete
        if status_np.sum():
            indices = numpy.where(status_np)
            raise Exception(
                'workers ' + ','.join(str(i) for i in indices) + ' failed')
        time.sleep(0.1)
    
    # save the policy and value
    np_policy = numpy.frombuffer(policy, dtype=numpy.float64).reshape(-1, 9)
    np_value = numpy.frombuffer(value, dtype=numpy.float64)
    with open('%s_final.pkl'%game_mode, 'wb') as f:
        pickle.dump({'p':np_policy, 'v':np_value}, f)

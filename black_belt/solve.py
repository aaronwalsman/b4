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

from black_belt.game import (
    PlayerState,
    State,
    game_mode,
    max_head_hits,
    max_body_hits,
    max_legs_hits,
    max_total_hits,
    start_head_a,
    start_head_ac,
    start_body_a,
    start_body_ac,
    start_legs_a,
    start_legs_ac,
)
from ne import lp_solve_zero_sum

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num-procs', type=int, default=40)

    args = parser.parse_args()

    # setup multiprocessing
    context = multiprocessing.get_context(None)
    policy_shape = (9, total_states)
    policy = context.RawArray('d', policy_shape[0]*policy_shape[1])
    value = context.RawArray('d', total_states)
    complete = context.RawArray('d', total_states)
    status = context.RawArray('d', args.num_procs)

    # zero/load previous data
    complete_np = numpy.frombuffer(complete, dtype=numpy.float64)
    numpy.copyto(complete_np, numpy.zeros(total_states))
    status_np = numpy.frombuffer(status, dtype=numpy.float64)
    numpy.copyto(status_np, numpy.zeros(args.num_procs))

    def payoff_matrix(index):
        state = index_to_state(index)
        p1_actions, p2_actions = state.action_space
        payoff = numpy.zeros((len(p2_actions), len(p1_actions)))
        for i, p1_action in enumerate(p1_actions):
            for j, p2_action in enumerate(p2_actions):
                successor = state.transition((p1_action, p2_action))
                successor_value = successor.value
                if successor_value is None:
                    successor_index = state_to_index(successor)
                    while not complete[successor_index]:
                        time.sleep(0.1)
                    successor_value = value[successor_index]
                payoff[j,i] = successor_value
        
        return payoff, p1_actions

    def worker(proc_id):
        try:
            for i in range(proc_id, total_states, args.num_procs):
                if complete[i]:
                    continue
                game, p1_actions = payoff_matrix(i)
                with warnings.catch_warnings():
                    warnings.simplefilter('error', OptimizeWarning)
                    try:
                        p, v = lp_solve_zero_sum(game)
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
                full_p[[int(a) for a in p1_actions]] = p
                policy[i*9:(i+1)*9] = full_p
                complete[i] = 1
        except:
            status[proc_id] = 1
            raise

    for i in range(args.num_procs):
        process = context.Process(
            target=worker,
            name='worker_%i'%i,
            args=(i,),
        )
        process.daemon = True
        process.start()

    progress = tqdm.tqdm(total=total_states+1)
    last_complete = 0
    total_complete = 0
    try:
        while total_complete < total_states:
            total_complete = complete_np.sum()
            progress.update(int(total_complete - last_complete))
            last_complete = total_complete
            if status_np.sum():
                indices = numpy.where(status_np)
                raise Exception(
                    'workers ' + ','.join(str(i) for i in indices) + ' failed')
            time.sleep(0.1)
            
    finally:
        pass

    np_policy = numpy.frombuffer(policy, dtype=numpy.float64).reshape(-1, 9)
    np_value = numpy.frombuffer(value, dtype=numpy.float64)

    with open('%s_final.pkl'%game_mode, 'wb') as f:
        pickle.dump({'p':np_policy, 'v':np_value}, f)

    def worker_OLD(start_cards, proc_id, num_procs):
        for num_cards in range(start_cards, total_starting_cards+1):
            print('process %i/%i is working on %i card games'%
                (proc_id, num_procs, num_cargs))
            
            if num_cards != 1:
                prev_status_path = status_path(num_cards-1)
                while not os.path.exists(prev_status_path):
                    print('process %i/%i is waiting for: %s'%prev_status_path)
                    time.sleep(1)
                prev_card_path = card_path(num_cards-1)
                previous_card_zip = ZipFile(prev_card_path)
            
            else:
                previous_card_zip = None
            
            with ZipFile(card_path(num_cards, proc_id), 'w') as progress_zip:
                n_card_states = [
                    cs for cs in card_states if cs.total == num_cards]
                total_steps = (
                    len(all_live_hit_states)**2 * len(n_card_states)**2)
                iterate = tqdm.tqdm(total=total_steps)
                with iterate:
                    for p1_hit_state in all_live_hit_states:
                        for p1_card_state in n_card_states:
                            p1 = PlayerState(
                                hit_state=p1_hit_state,
                                card_state=p1_card_state,
                            )
                            for p2_hit_state in all_live_hit_states:
                                for p2_card_state in n_card_states:
                                    if iterate.n % num_procs != proc_id:
                                        iterate.update(1)
                                        continue
                                    
                                    p2 = PlayerState(
                                        hit_state=p2_hit_state,
                                        card_state=p2_card_state,
                                    )
                                    state = State(p1, p2)
                                    payoff = payoff_matrix(state, previous_card_zip)
                                    try:
                                        policy, value = linear_zero_sum(payoff)
                                    except ValueError:
                                        numpy.save('./dump_%i.npy'%iterate.n, payoff)
                                        raise
                                    #state_policies[state] = policy
                                    #state_values[state] = value
                                    policy_path = state.serialize() + '_policy.npy'
                                    with progress_zip.open(policy_path, 'w') as f:
                                        numpy.save(f, policy)
                                    
                                    value_path = state.serialize() + '_value.npy'
                                    with progress_zip.open(value_path, 'w') as f:
                                        numpy.save(f, value)
                                    
                                    iterate.update(1)
            
            with open(status_path(num_cards, proc_id), 'w') as f:
                f.write('finished')
            
            if proc_id == 0:
                path = card_path(num_cards)
                with ZipFile(path, 'w') as combine_zip:
                    for i in range(num_procs):
                        wait_path = status_path(num_cards, i)
                        while not os.path.exists(wait_path):
                            print('Waiting for: %s'%wait_path)
                            time.sleep(1)
                        progress_path = card_path(num_cards, i)
                        with ZipFile(progress_path, 'r') as progress_zip:
                            for info in progress_zip.infolist():
                                with progress_zip.open(info.filename, 'r') as fr:
                                    with combine_zip.open(info.filename, 'w') as fw:
                                        fw.write(fr.read())
                        
                        os.remove(progress_path)
                        os.remove(wait_path)
                
                with open(status_path(num_cards), 'w') as f:
                    f.write('finished')

import time
import os
import sys
import multiprocessing
import pickle
from zipfile import ZipFile
from argparse import ArgumentParser

import numpy

import tqdm

from black_belt.game import (
    HitState,
    CardState,
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

def payoff_matrix_OLD(state, previous_card_zipfile):
    NO
    p1_actions, p2_actions = state.action_space()
    p1_actions = p1_actions
    p2_actions = p2_actions
    payoff = numpy.zeros((len(p2_actions), len(p1_actions)))
    for i, p1_action in enumerate(p1_actions):
        for j, p2_action in enumerate(p2_actions):
            successor = state.transition((p1_action, p2_action))
            value = successor.value()
            if value is None:
                #value = state_values[successor]
                value_path = successor.serialize() + '_value.npy'
                with previous_card_zipfile.open(value_path) as value_file:
                    value = numpy.load(value_file)
            payoff[j,i] = value
    
    return payoff

'''
def card_path(num_cards, proc_id=None):
    if proc_id is None:
        return './data/%s_%i.zip'%(game_mode, num_cards)
    else:
        return './data/%s_%i_%i.zip'%(game_mode, num_cards, proc_id)

def status_path(num_cards, proc_id=None):
    if proc_id is None:
        return './data/%s_%i.status'%(game_mode, num_cards)
    else:
        return './data/%s_%i_%i.status'%(game_mode, num_cards, proc_id)
'''

'''
def compute_ordered_states():
    
    ordered_states = numpy.zeros((total_states, 18))
    breakpoint()
    for i in range(1, total_starting_cards+1):
        print('computing all states with %i cards'%i)
        for p1_card_state in card_states[i]:
            for p2_card_state in card_states[i]:
                for p1_hit_state in all_live_hit_states:
                    for p2_hit_state in all_live_hit_states:
                        p1_state = PlayerState(p1_hit_state, p1_card_state)
                        p2_state = PlayerState(p2_hit_state, p2_card_state)
                        game_state = State(p1_state, p2_state)
                        breakpoint()
                        ordered_states.append(game_state)
    
    return ordered_states
'''

parser = ArgumentParser()
parser.add_argument('--num-procs', type=int, default=40)

args = parser.parse_args()

all_live_hit_states = []
for h in range(max_head_hits):
    for b in range(max_body_hits):
        for l in range(max_legs_hits):
            hit_state = HitState(h,b,l)
            if not hit_state.is_dead:
                all_live_hit_states.append(hit_state)

card_states = {}
for ha in range(start_head_a+1):
    for hac in range(start_head_ac+1):
        for ba in range(start_body_a+1):
            for bac in range(start_body_ac+1):
                for la in range(start_legs_a+1):
                    for lac in range(start_legs_ac+1):
                        card_state = CardState(ha, hac, ba, bac, la, lac)
                        if card_state.total not in card_states:
                            card_states[card_state.total] = []
                        card_states[card_state.total].append(card_state)

total_starting_cards = max(card_states.keys())

total_states = 0
card_count_ranges = {}
for i in range(1, total_starting_cards+1):
    start = total_states
    total_states += len(card_states[i])**2*len(all_live_hit_states)**2
    card_count_ranges[i] = start, total_states

def state_to_index(state):
    p1_cards = state.p1.card_state.total
    p2_cards = state.p2.card_state.total
    assert p1_cards == p2_cards
    range_start, range_end = card_count_ranges[p1_cards]
    
    p1_c = card_states[p1_cards].index(state.p1.card_state)
    p2_c = card_states[p1_cards].index(state.p2.card_state)
    p1_h = all_live_hit_states.index(state.p1.hit_state)
    p2_h = all_live_hit_states.index(state.p2.hit_state)
    
    c = len(card_states[p1_cards])
    h = len(all_live_hit_states)
    
    index = numpy.ravel_multi_index((p1_c, p2_c, p1_h, p2_h), (c, c, h, h))
    index += range_start
    assert index < range_end
    return index

def index_to_state(index):
    num_cards = 1
    while index < card_count_ranges[num_cards][0]:
        num_cards += 1
    
    range_start, range_end = card_count_ranges[num_cards]
    
    c = len(card_states[num_cards])
    h = len(all_live_hit_states)
    p1_c, p2_c, p1_h, p2_h = numpy.unravel_index(
        index - range_start, (c, c, h, h))
    
    p1 = PlayerState(
        all_live_hit_states[p1_h],
        card_states[num_cards][p1_c],
    )
    p2 = PlayerState(
        all_live_hit_states[p2_h],
        card_states[num_cards][p2_c],
    )
    return State(p1, p2)

context = multiprocessing.get_context(None)
policy_shape = (9, total_states)
p1_policy = context.RawArray('d', policy_shape[0]*policy_shape[1])
p2_policy = context.RawArray('d', policy_shape[0]*policy_shape[1])
value = context.RawArray('d', total_states)
complete = context.RawArray('d', total_states)
complete_np = numpy.frombuffer(complete, dtype=numpy.float64)
numpy.copyto(complete_np, numpy.zeros(total_states))

def payoff_matrix(index):
    state = index_to_state(index)
    p1_actions, p2_actions = state.action_space()
    payoff = numpy.zeros((len(p2_actions), len(p1_actions)))
    for i, p1_action in enumerate(p1_actions):
        for j, p2_action in enumerate(p2_actions):
            successor = state.transition((p1_action, p2_action))
            successor_value = successor.value()
            if value is None:
                successor_index = state_to_index(successor)
                while not complete[successor_index]:
                    time.sleep(0.1)
                successor_value = value[successor_index]
            payoff[j,i] = successor_value
    
    return payoff

def worker(proc_id):
    for i in range(proc_id, total_states, args.num_procs):
        state = index_to_state(i)
        CONTINUE_HERE
        complete[i] = 1

for i in range(args.num_procs):
    process = context.Process(
        target=worker,
        name='worker_%i'%i,
        args=(i,),
    )
    process.daemon = True
    process.start()

while complete_np.sum() != total_states:
    print(complete_np.sum() / total_states)

breakpoint()

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
            n_card_states = [cs for cs in card_states if cs.total == num_cards]
            total_steps = len(all_live_hit_states)**2 * len(n_card_states)**2
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

with open('%s_final.pkl'%game_mode, 'wb') as f:
    pickle.dump({'p':state_policies, 'v':state_values}, f)

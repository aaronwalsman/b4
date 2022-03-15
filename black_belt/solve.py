import time
import os
import sys
from zipfile import ZipFile

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
from ne import equilibrium

all_live_hit_states = []
for h in range(max_head_hits):
    for b in range(max_body_hits):
        for l in range(max_legs_hits):
            hit_state = HitState(h,b,l)
            if not hit_state.is_dead():
                all_live_hit_states.append(hit_state)

card_states = []
for ha in range(start_head_a+1):
    for hac in range(start_head_ac+1):
        for ba in range(start_body_a+1):
            for bac in range(start_body_ac+1):
                for la in range(start_legs_a+1):
                    for lac in range(start_legs_ac+1):
                        card_state = CardState(
                            ha, hac, ba, bac, la, lac)
                        card_states.append(card_state)

def payoff_matrix(state, previous_card_zipfile):
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

total_starting_cards = (
    start_head_a + start_head_ac +
    start_body_a + start_body_ac +
    start_legs_a + start_legs_ac
)

start_cards = int(sys.argv[-2])
rank = int(sys.argv[-1])
size = 8

def card_path(num_cards, rank=None):
    if rank is None:
        return './data/%s_%i.zip'%(game_mode, num_cards)
    else:
        return './data/%s_%i_%i.zip'%(game_mode, num_cards, rank)

def status_path(num_cards, rank=None):
    if rank is None:
        return './data/%s_%i.status'%(game_mode, num_cards)
    else:
        return './data/%s_%i_%i.status'%(game_mode, num_cards, rank)

for num_cards in range(start_cards, total_starting_cards+1):
    print('='*80)
    print('Solving %i Card Games'%num_cards)

    if num_cards != 1:
        prev_status_path = status_path(num_cards-1)
        while not os.path.exists(prev_status_path):
            print('Waiting for: %s'%prev_status_path)
            time.sleep(1)
        prev_card_path = card_path(num_cards-1)
        previous_card_zip = ZipFile(prev_card_path)
    
    else:
        previous_card_zip = None
    
    with ZipFile(card_path(num_cards, rank), 'w') as progress_zip:
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
                            if iterate.n % size != rank:
                                iterate.update(1)
                                continue
                            
                            p2 = PlayerState(
                                hit_state=p2_hit_state,
                                card_state=p2_card_state,
                            )
                            state = State(p1, p2)
                            payoff = payoff_matrix(state, previous_card_zip)
                            try:
                                policy, value = equilibrium(payoff)
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
    
    with open(status_path(num_cards, rank), 'w') as f:
        f.write('finished')
    
    if rank == 0:
        path = card_path(num_cards)
        with ZipFile(path, 'w') as combine_zip:
            for i in range(size):
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

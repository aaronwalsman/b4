import pickle

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
from ne import ne

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

state_policies = {}
state_values = {}

def payoff_matrix(state):
    p1_actions, p2_actions = state.action_space()
    p1_actions = p1_actions
    p2_actions = p2_actions
    payoff = numpy.zeros((len(p2_actions), len(p1_actions)))
    for i, p1_action in enumerate(p1_actions):
        for j, p2_action in enumerate(p2_actions):
            successor = state.transition((p1_action, p2_action))
            value = successor.value()
            if value is None:
                value = state_values[successor]
            payoff[j,i] = value
    
    return payoff

total_starting_cards = (
    start_head_a + start_head_ac +
    start_body_a + start_body_ac +
    start_legs_a + start_legs_ac
)

start_cards = 1
rank = 0
size = 8

if start_cards != 1:
    print('Loading old data')
    with open('%s_wip_%i.pkl'%(game_mode, start_cards-1), 'rb') as f:
        data = pickle.load(f)
        state_policies.update(data['p'])
        state_values.update(data['v'])
    print('Loading complete')

for num_cards in range(start_cards, total_starting_cards+1):
    print('='*80)
    print('Solving %i Card Games'%num_cards)
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
                        p2 = PlayerState(
                            hit_state=p2_hit_state,
                            card_state=p2_card_state,
                        )
                        state = State(p1, p2)
                        payoff = payoff_matrix(state)
                        policy, value = ne(payoff)
                        state_policies[state] = policy
                        state_values[state] = value
                        iterate.update(1)

    with open('%s_wip_%i.pkl'%(game_mode, num_cards), 'wb') as f:
        pickle.dump({'p':state_policies, 'v':state_values}, f)

with open('%s_final.pkl'%game_mode, 'wb') as f:
    pickle.dump({'p':state_policies, 'v':state_values}, f)

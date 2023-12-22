import time

import numpy

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

'''
This module compiles a list of all possible live hit states and card states
so that the solver can iterate through them.
'''

# compute all live HitStates
all_live_hit_states = []
for h in range(max_head_hits):
    for b in range(max_body_hits):
        for l in range(max_legs_hits):
            hit_state = HitState(h,b,l)
            if not hit_state.is_dead:
                all_live_hit_states.append(hit_state)

# compute all possible CardStates
# these are stored in a dictionary mapping the number of cards remaining to
# all possible CardStates with that many cards
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

# compute the number of cards that a player starts with
total_starting_cards = max(card_states.keys())

# Make a lookup that maps a State to an integer index so that a policy can be
# represented as one giant numpy array.  The first step is to make a set of
# ranges that describe where the CardStates with different numbers of cards
# will reside in the final index space.
total_states = 0
card_count_ranges = {}
for i in range(1, total_starting_cards+1):
    start = total_states
    total_states += len(card_states[i])**2*len(all_live_hit_states)**2
    card_count_ranges[i] = start, total_states

# maps a State to an integer index
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

# maps an integer index to a State
def index_to_state(index):
    '''
    Maps an integer index to a game state.  States with more cards are indexed
    with smaller values than those with more cards.
    '''
    for num_cards, (range_start, range_end) in card_count_ranges.items():
        if index >= range_start and index < range_end:
            break
    else:
        raise IndexError('index too large')

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

# generates a payoff matrix for a particular state using known values of all
# possible successor states
def payoff_matrix(state, value, complete=None):
    p1_actions, p2_actions = state.action_space
    payoff = numpy.zeros((len(p2_actions), len(p1_actions)))
    for i, p1_action in enumerate(p1_actions):
        for j, p2_action in enumerate(p2_actions):
            successor = state.transition((p1_action, p2_action))
            successor_value = successor.value
            if successor_value is None:
                successor_index = state_to_index(successor)
                if complete is not None:
                    while not complete[successor_index]:
                        time.sleep(0.1)
                successor_value = value[successor_index]
            payoff[j,i] = successor_value

    return payoff, p1_actions

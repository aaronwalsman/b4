import random
import pickle

import numpy

from black_belt.bodega_brawl import State, Action, action_order
from black_belt.game_statistics import state_to_index, payoff_matrix
from black_belt.ne import best_response

class Agent:
    def play(self, state):
        my_actions, opponent_actions = state.action_space
        policy = self.policy(state)
        weights = [policy[int(a)] for a in my_actions]
        return random.choices(my_actions, weights=weights)[0]

class SolvedAgent(Agent):
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def policy(self, state):
        index = state_to_index(state)
        policy = self.data['p'][index]
        return policy
    
    def value(self, state):
        index = state_to_index(state)
        value = self.data['v'][index]
        return value

class RandomAgent(Agent):
    def policy(self, state):
        my_actions, _ = state.action_space
        policy = [0.] * 9
        for a in my_actions:
            policy[int(a)] = 1./len(my_actions)
        return policy

class BestResponseAgent(Agent):
    def __init__(self, opponent):
        self.opponent = opponent
    
    def policy(self, state):
        my_actions, opponent_actions = state.action_space
        opposite_state = State(state.p2, state.p1)
        game, _ = payoff_matrix(state, self.opponent.data['v'])
        opponent_policy = self.opponent.policy(opposite_state)
        partial_opponent_policy = opponent_policy[
            [int(a) for a in opponent_actions]]
        partial_policy, v = best_response(game, partial_opponent_policy)
        policy = [0.] * 9
        for i,a in enumerate(my_actions):
            policy[int(a)] = partial_policy[i]
        return policy

class ArgmaxCounterAgent(Agent):
    def __init__(self, opponent):
        self.opponent = opponent
        self.random_agent = RandomAgent()
    
    def policy(self, state):
        my_actions, opponent_actions = state.action_space
        opposite_state = State(state.p2, state.p1)
        opponent_policy = self.opponent.play(opposite_state)
        opponent_index = numpy.argmax(opponent_policy)
        opponent_action = action_order[opponent_index]
        if opponent_action.mode == 'attack':
            action = Action(opponent_action.region, 'attack/counter', 'counter')
            if action in my_actions:
                policy = [0.] * 9
                policy[int(action)] = 1.
                return policy
        
        region_priorities = sorted([
            (2 - state.p2.hit_state.head, 'head'),
            (3 - state.p2.hit_state.body, 'body'),
            (4 - state.p2.hit_state.legs, 'legs')
        ])
        for hits_remaining, region in region_priorities:
            card_state = state.p1.card_state
            a_left = getattr(card_state, '%s_a'%region)
            ac_left = getattr(card_state, '%s_ac'%region)
            if a_left + ac_left >= hits_remaining:
                if a_left:
                    action = Action(region, 'attack', 'attack')
                else:
                    action = Action(region, 'attack/counter', 'attack')
                policy = [0.] * 9
                policy[int(action)] = 1.
                return policy
        
        return self.random_agent.policy(state)

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

class MCTSNode:
    def __init__(self, state, c=2**0.5, child_nodes=None):
        self.state = state
        self.c = c
        self.visits = 0
        self.my_actions, self.opponent_actions = state.action_space
        self.my_child_visits = numpy.zeros(len(self.my_actions))
        self.my_child_values = numpy.zeros(len(self.my_actions))
        self.opponent_child_visits = numpy.zeros(len(self.opponent_actions))
        self.opponent_child_values = numpy.zeros(len(self.opponent_actions))
        self.opponent_visits = 0
        if child_nodes is None:
            child_nodes = {}
        self.child_nodes = child_nodes
        self.random_agent = RandomAgent()
        
    def sample(self):
        my_unsampled_children = numpy.where(self.my_child_visits == 0)[0]
        if my_unsampled_children.shape[0]:
            my_child_index = random.choice(my_unsampled_children)
            my_action = self.my_actions[my_child_index]
        else:
            my_ucb = (
                (self.my_child_values / self.my_child_visits) +
                self.c * (numpy.log(self.visits)/self.my_child_visits)
            )
            my_child_index = numpy.argmax(my_ucb)
            my_action = self.my_actions[my_child_index]
        self.my_child_visits[my_child_index] += 1
            
        opponent_unsampled_children = numpy.where(
            self.opponent_child_visits == 0)[0]
        if opponent_unsampled_children.shape[0]:
            opponent_child_index = random.choice(opponent_unsampled_children)
            opponent_action = self.opponent_actions[opponent_child_index]
        else:
            opponent_ucb = (
                (1.-self.opponent_child_values / self.opponent_child_visits) +
                self.c * (numpy.log(self.visits)/self.opponent_child_visits)
            )
            opponent_child_index = numpy.argmax(opponent_ucb)
            opponent_action = self.opponent_actions[opponent_child_index]
        self.opponent_child_visits[opponent_child_index] += 1
        self.visits += 1
        
        next_state = self.state.transition((my_action, opponent_action))
        if next_state.terminal:
            value = next_state.value
        
        elif next_state in self.child_nodes:
            value, _, _ = self.child_nodes[next_state].sample()
        
        else:
            self.child_nodes[next_state] = MCTSNode(
                next_state, c=self.c, child_nodes=self.child_nodes)
            while not next_state.terminal:
                my_next_actions, opponent_next_actions = next_state.action_space
                my_next_action = random.choice(my_next_actions)
                opponent_next_action = random.choice(opponent_next_actions)
                next_state = next_state.transition(
                    (my_next_action, opponent_next_action))
            
            value = next_state.value
        
        self.my_child_values[my_child_index] += value
        self.opponent_child_values[opponent_child_index] += value
        
        return value, my_action, opponent_action

class MCTSAgent(Agent):
    def __init__(self, samples=10000):
        self.samples = samples
        self.mcts_nodes = {}
        self.root_node = MCTSNode(State(), child_nodes=self.mcts_nodes)
        self.mcts_nodes[State()] = self.root_node
    
    def policy(self, state):
        if state not in self.mcts_nodes:
            self.mcts_nodes[state] = MCTSNode(
                state, child_nodes=self.mcts_nodes)
        mcts_node = self.mcts_nodes[state]
        for i in range(self.samples):
            mcts_node.sample()
        v, my_action, _ = mcts_node.sample()
        my_policy = [0.] * 9
        my_policy[int(my_action)] = 1.
        return my_policy

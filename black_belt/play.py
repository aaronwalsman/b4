import random
import pickle

from game import State, game_mode

class SolvedAgent:
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
    
    def play(self, state):
        policy = self.data['p'][state]
        return policy

agent = SolvedAgent('./%s_final.pkl'%game_mode)

state = State()
while not state.terminal():
    p1_actions, p2_actions = state.action_space()
    
    print(state)
    
    # pick an action for player 1
    print('Opponent played the following distribution:')
    agent_policy = agent.play(state)
    for i, p1_action in enumerate(p1_actions):
        print('%i: %s [%.04f]'%(i, p1_action, agent_policy[i]))
    print('V: %.04f'%agent.data['v'][state])
    print()
    
    opposite_state = State(state.p2, state.p1)
    recommended_policy = agent.play(opposite_state)
    # pick an action for player 2
    print('Choose an action:')
    for i, p2_action in enumerate(p2_actions):
        print('%i: %s [%.04f]'%(i, p2_action, recommended_policy[i]))
    i = int(input())
    p2_action = p2_actions[i]
    
    p1_action = random.choices(p1_actions, weights=agent_policy)[0]
    print('Opponent sampled: %s'%str(p1_action))
    
    state = state.transition((p1_action, p2_action))

print('Final State:')
print(state)
print(state.value())

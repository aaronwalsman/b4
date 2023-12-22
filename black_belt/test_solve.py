import random
import pickle
import argparse

import tqdm

from black_belt.bodega_brawl import State, game_mode
from black_belt.game_statistics import state_to_index, payoff_matrix
from black_belt.agent import SolvedAgent
from black_belt.ne import best_response

parser = argparse.ArgumentParser()
parser.add_argument('--opponent', type=str, default='random')
parser.add_argument('--games', type=int, default=10000)

def test_solve():
    agent = SolvedAgent('./solutions/%s_final.pkl'%game_mode)

    opponent = 'random'

    n_games = 10000

    results = []
    for i in tqdm.tqdm(range(n_games)):
        state = State()
        while not state.terminal:
            p1_actions, p2_actions = state.action_space
            
            # pick an action for player 1
            state_index = state_to_index(state)
            agent_policy = agent.play(state_index)
            weights = [agent_policy[int(a)] for a in p1_actions]
            p1_action = random.choices(p1_actions, weights=weights)[0]
            
            if opponent == 'random':
                p2_action = random.choice(p2_actions)
            elif opponent == 'best_response':
                opposite_state = State(state.p2, state.p1)
                game, _ = payoff_matrix(opposite_state, agent.data['v'])
                partial_agent_policy = agent_policy[
                    [int(a) for a in p1_actions]]
                opponent_policy, v = best_response(game, partial_agent_policy)
                p2_action = random.choices(
                    p2_actions, weights=opponent_policy)[0]
            
            state = state.transition((p1_action, p2_action))
        
        results.append(state.value)

    average_value = (sum(results)/len(results) - 0.5) * (1./0.8) + 0.5
    print(average_value)

if __name__ == '__main__':
    test_solve()

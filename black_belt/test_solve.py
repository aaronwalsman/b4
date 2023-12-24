import random
import pickle
import argparse

import tqdm

from black_belt.bodega_brawl import State, game_mode
from black_belt.game_statistics import payoff_matrix
from black_belt.agent import (
    SolvedAgent,
    RandomAgent,
    BestResponseAgent,
    ArgmaxCounterAgent,
)
from black_belt.ne import best_response

parser = argparse.ArgumentParser()
parser.add_argument('--opponent', type=str, default='random')
parser.add_argument('--games', type=int, default=10000)

def test_solve(opponent='random', games=10000):
    
    # load the agent
    agent_path = './solutions/%s_final.pkl'%game_mode
    agent = SolvedAgent(agent_path)
    
    # load the opponent
    if opponent == 'random':
        opponent_agent = RandomAgent()
    elif opponent == 'best_response':
        opponent_agent = BestResponseAgent(agent)
    elif opponent == 'argmax_counter':
        opponent_agent = ArgmaxCounterAgent(agent)
    elif opponent == 'solved':
        opponent_agent = SolvedAgent(agent_path)
    
    results = []
    for i in tqdm.tqdm(range(games)):
        state = State()
        while not state.terminal:
            
            # pick an action for the agent
            p1_action = agent.play(state)
            
            # pick an action for the opponent
            opposite_state = State(state.p2, state.p1)
            p2_action = opponent_agent.play(opposite_state)
            
            state = state.transition((p1_action, p2_action))
        
        results.append(state.value)

    average_value = (sum(results)/len(results) - 0.5) * (1./0.8) + 0.5
    print('Win Rate: %.06f'%average_value)
    
    wins = sum([r > 0.6 for r in results])
    losses = sum([r < 0.4 for r in results])
    draws = len(results) - wins - losses
    print('Wins: %i, Draws: %i, Losses: %i'%(wins, draws, losses))

def test_solve_commandline():
    # parse args
    args = parser.parse_args()
    
    test_solve(opponent=args.opponent, games=args.games)

if __name__ == '__main__':
    test_solve_from_args()

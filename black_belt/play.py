import random
import pickle
import argparse

from black_belt.bodega_brawl import State, game_mode
from black_belt.game_statistics import state_to_index
from black_belt.agent import SolvedAgent

'''
Interactively play against the computer.  Two flags:
--drive : When this flag is set, the script will show the computer's moves
before asking for the player input.  This is useful for when one human is
using this script to play the game against another human using physical cards.
--verbose : Will show the computer's action probabilities and value estimates
when playing.
'''

parser = argparse.ArgumentParser()
parser.add_argument(
    '--drive', action='store_true',
    help='For use when playing against another human with physical cards.')
parser.add_argument(
    '--verbose', action='store_true',
    help='Shows action probabilities and value estimates')

def play(drive=False, verbose=False):
    
    # load agent
    agent = SolvedAgent('./solutions/%s_final.pkl'%game_mode)
    
    # initialize the game state
    state = State()
    
    # continue until terminal
    while not state.terminal:
        # print the current state
        print(str(state).replace('p1: ', 'cpu:').replace('p2: ', 'you:'))
        
        # get the available actions
        p1_actions, p2_actions = state.action_space
        
        # look up the agent's policy
        agent_policy = agent.policy(state)
        if verbose:
            print('CPU played the following distribution:')
            for i, p1_action in enumerate(p1_actions):
                print('%i: %s [%.04f]'%(
                    i, p1_action, agent_policy[int(p1_action)]))
            print('V: %.04f'%agent.value(state))
            print()
        
        # if this is in drive mode, pick a CPU action now
        if drive:
            p1_action = agent.play(state)
            print('CPU played: %s'%str(p1_action))
            print()
        
        # get the recommended policy for the human
        opposite_state = State(state.p2, state.p1)
        recommended_policy = agent.policy(opposite_state)
        
        # pick an action for player 2
        print('Choose an action:')
        for i, p2_action in enumerate(p2_actions):
            if verbose:
                print('%i: %s [%.04f]'%(
                    i, p2_action, recommended_policy[int(p2_action)]))
            else:
                print('%i: %s'%(i, p2_action))
        if verbose:
            print('V: %0.4f'%agent.data['v'][opposite_state_index])
            print()
        
        # get the user's input
        print('Select an index:')
        i = int(input())
        p2_action = p2_actions[i]
        
        # if not in drive mode, pick a CPU action now
        if not drive:
            p1_action = agent.play(state)
            print('CPU played: %s'%str(p1_action))
        
        # transition to the next state using the players' actions
        state = state.transition((p1_action, p2_action))
    
    # print the final state
    print('Final State:')
    print(str(state).replace('p1: ', 'cpu:').replace('p2: ', 'you:'))
    print(state.value)

def play_commandline():
    # parse args
    args = parser.parse_args()
    
    play(drive=args.drive, verbose=args.verbose)

if __name__ == '__main__':
    play_commandline()

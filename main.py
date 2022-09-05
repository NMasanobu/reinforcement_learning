from copy import deepcopy
import json
import argparse

import numpy as np

from game.tic_tac_toe import TicTacToe
from model_based.environment import T3Environment
from model_based.planner import ValueIterationPlanner

def train():
    env = T3Environment()
    planner = ValueIterationPlanner(env)
    V = planner.plan()

    with open('output/model_based_vi.json', mode='w') as f:
        json.dump(V, f)

def play():
    game = TicTacToe()
    env = T3Environment()

    with open('output/model_based_vi.json') as f:
        V = json.load(f)
    
    n_episodes = 1000

    # result
    # 0: player first win
    # 1: player first lose
    # 2: player first draw
    # 3: cpu first win
    # 4: cpu first lose
    # 5: cpu first draw
    # 9: others
    results = np.zeros(n_episodes)

    for e in range(n_episodes):
        game.reset()
        game.start()
        game_settings = game.get_game_settings()
        player_first = game_settings['player_first']

        state_type = 0
        while state_type == 0:
            available_actions = game.get_available_actions()

            value_list = [] # order coresponds that of avaliable_actions
            for a in available_actions:
                next_state_list = []
                state_str = env.matrix2state(game.matrix)
                state_list = list(state_str)

                # apply action
                state_list[a] = '1'

                if not '0' in state_list:
                    # all cells are filled
                    next_state_list.append(''.join(state_list))

                else:
                    # posible states after cpu action
                    for i in range(9):
                        if state_list[i] == '0':
                            state_list_copy = deepcopy(state_list)
                            state_list_copy[i] = '2'
                            next_state_list.append(''.join(state_list_copy))

                # maximize expected value
                value_list.append(np.mean([V[s] for s in next_state_list]))

            # maximize expected value
            action = available_actions[np.argmax(value_list)]
            state_type = game.step(action)

        if state_type == 1:
            if player_first:
                results[e] = 0
            else:
                results[e] = 3

        elif state_type == 2:
            if player_first:
                results[e] = 1
            else:
                results[e] = 4

        elif state_type == 3:
            if player_first:
                results[e] = 2
            else:
                results[e] = 5

        elif state_type == 9:
            results[e] = 9
            
    n_player_first_win = (results==0).sum()
    n_player_first_lose = (results==1).sum()
    n_player_first_draw = (results==2).sum()
    n_cpu_first_win = (results==3).sum()
    n_cpu_first_lose = (results==4).sum()
    n_cpu_first_draw = (results==5).sum()
    others = (results==9).sum()

    print('results:')
    print(f'player first win: {n_player_first_win}')
    print(f'player first lose: {n_player_first_lose}')
    print(f'player first draw: {n_player_first_draw}')
    print(f'cpu first win: {n_cpu_first_win}')
    print(f'cpu first lose: {n_cpu_first_lose}')
    print(f'cpu first draw: {n_cpu_first_draw}')
    print(f'others: {others}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help='If execute train method.')
    parser.add_argument("--play", action='store_true', help='If execute play method.')

    args = parser.parse_args()
    require_train = args.train
    require_play = args.play

    if not (require_train or require_play):
        parser.print_help()
    
    if require_train:
        train()

    if require_play:
        play()
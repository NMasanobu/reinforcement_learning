import sys
import itertools
from copy import deepcopy

import numpy as np

# for debugging
from game.tic_tac_toe import TicTacToe

class T3Environment():
    def __init__(self):
        self.t3_game = TicTacToe()

        self.actions = list(range(9))

        self.states = []
        for s in itertools.product('012', repeat=9):
            num_0 = s.count('0')
            num_1 = s.count('1')
            num_2 = s.count('2')

            # ignore invalid states (corresponds of number of player action and that of cpu acition is invalid)
            if abs(num_1 - num_2) > 1:
                continue

            # ignore states only cpu can take action (number of player action = number of cpu action + 1)
            # but end states (no zeros) will be validated
            elif num_1 - num_2 > 0 and num_0 > 0:
                continue

            self.states.append(''.join(s))

        self.reset()
    
    def reset(self):
        self.t3_game.reset()

    def game_start(self):
        self.t3_game.start()

    def check_state(self, state):
        '''
        check current state:
            0: continue game
            1: win
            2: lose
            3: draw
            9: unexpected state
        '''

        state_type = 0
        win = False
        lose = False

        matrix = self.state2matrix(state)

        sum_v = matrix.sum(0)
        sum_h = matrix.sum(1)
        sum_d = [ # diag
            matrix[0, 0] + matrix[1, 1] + matrix[2, 2],
            matrix[0, 2] + matrix[1, 1] + matrix[2, 0]
        ]
        sum_vhd = np.hstack([sum_v, sum_h, sum_d])

        if 3 in sum_vhd:
            win = True

        if -3 in sum_vhd:
            lose = True

        if win and lose:
            state_type = 9
        elif win:
            state_type = 1
        elif lose:
            state_type = 2
        else:
            if (matrix==0).sum() == 0:
                state_type = 3

        return state_type

    def is_valid_action(self, state, action):
        matrix_1d = self.state2matrix(state).reshape(9)

        is_valid = True
        if matrix_1d[action] != 0:
            # a selected cell is already filled
            is_valid = False

        return is_valid

    def get_available_actions_at(self, state):
        actions = []
        for a in range(9):
            if state[a] == '0':
                actions.append(a)

        return actions

    def transit_func(self, state, action):
        '''
        returns:
            (dict): key: state id, value: transition prob
        '''
        state_list = list(state)
        state_list[action] = '1'

        next_state_list = []
        for i in range(9):
            if state_list[i] == '0':
                state_list_copy = deepcopy(state_list)
                state_list_copy[i] = '2'
                next_state_list.append(''.join(state_list_copy))
            
        transition_dict = {next_state: 1/len(next_state_list) for next_state in next_state_list}

        return transition_dict

    def reward_func(self, state):
        reward = 0.
        state_type = self.check_state(state)

        num_0 = state.count('0')

        # calculate reward only at game end state.
        if num_0 == 0:
            num_1 = state.count('1')
            num_2 = state.count('2')

            player_first = True
            if num_1 < num_2:
                player_first = False

            if player_first:
                if state_type == 1:
                    reward = 2.
                elif state_type == 2:
                    reward = -10.
                elif state_type == 3:
                    reward = -2.

            else:
                if state_type == 1:
                    reward = 10.
                elif state_type == 2:
                    reward = -10.
                elif state_type == 3:
                    reward = 10.

        return reward

    def state2matrix(self, state):
        '''
        convert state id to 2d-array

        args:
            state (str): state id, 9 digits string
        '''

        # reshape to 2d-array (3x3) later.
        matrix = np.zeros(9).astype(int)
        for i, s in enumerate(state):
            # if s == '0', matrix[i] == 0 <- can skip because of default value 0.
            if s == '1':
                matrix[i] = 1
            elif s == '2':
                matrix[i] = -1

        return matrix.reshape((3, 3))

    def matrix2state(self, matrix):
        matrix = matrix.reshape(9).astype(int)
        matrix[matrix==-1] = 2
        state = ''.join([str(s) for s in matrix])

        return state

if __name__ == '__main__':
    env = T3Environment()
    state = '000000000'
    print(env.check_state(state))
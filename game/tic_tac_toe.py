import numpy as np

class TicTacToe():
    def __init__(self, use_logger=False, player_first=None):
        self.use_logger = use_logger
        self.player_first = player_first
        self.reset()

    def reset(self):
        self.matrix = np.zeros((3, 3))
        self.available_actions = list(range(9))
        if self.player_first is None:
            self.player_first = np.random.randint() % 2 == 0
        self.cpu = None

    def start(self):
        if not self.player_first:
            self.cpu_action()

    def set_cpu(self, cpu):
        self.cpu = cpu

    def get_available_actions(self):
        return self.available_actions

    def step(self, action):
        '''
        returns: state after cpu action, following player action
        '''
        # player action
        coord = self.action2coord(action)

        if self.matrix[coord[0], coord[1]] != 0:
            raise ValueError(f'Invalid action: {action}')

        self.matrix[coord[0], coord[1]] = 1
        self.available_actions.remove(action)

        if self.use_logger:
            print(self.matrix)

        state = self.check_state()
        if state != 0:
            return state

        # cpu action
        self.cpu_action()
        state = self.check_state()
        if state != 0:
            return state

        # if reach here, state is 0
        return state

    def cpu_action(self):
        if self.cpu is None:
            action = np.random.choice(self.available_actions)
            coord = self.action2coord(action)
        
        else:
            raise NotImplementedError()

        self.matrix[coord[0], coord[1]] = -1
        self.available_actions.remove(action)

        if self.use_logger:
            print(self.matrix)

    def check_state(self):
        '''
        return current state:
            0: continue game
            1: win
            2: lose
            3: draw
            9: unexpected state
        '''

        state = 0
        win = False
        lose = False

        sum_v = self.matrix.sum(0)
        sum_h = self.matrix.sum(1)
        sum_d = [
            game.matrix[0, 0] + game.matrix[1, 1] + game.matrix[2, 2],
            game.matrix[0, 2] + game.matrix[1, 1] + game.matrix[2, 0]
        ]
        sum_0_1 = np.hstack([sum_v, sum_h, sum_d])

        if 3 in sum_0_1:
            win = True

        if -3 in sum_0_1:
            lose = True

        if win and lose:
            state = 9
        elif win:
            state = 1
        elif lose:
            state = 2
        else:
            if len(self.available_actions) == 0:
                state = 3

        return state

    def action2coord(self, action):
        '''
        convert action ID to coordinate
        '''

        return (action // 3, action % 3)

    def play(self):
        done = False
        while done:
            done = True

if __name__ == '__main__':
    game = TicTacToe(use_logger=True, player_first=True)

    state = 0
    while state == 0:
        available_actions = game.get_available_actions()
        action = np.random.choice(available_actions)
        state = game.step(action)

    if state == 1:
        print('You win!')
    elif state == 2:
        print('You lose...')
    elif state == 3:
        print('Draw.')
    elif state == 9:
        print('No contest.')
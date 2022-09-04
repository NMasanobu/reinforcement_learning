import numpy as np

class TicTacToe():
    def __init__(self, use_logger=False, player_first=None):
        self.use_logger = use_logger
        self.player_first = player_first
        if player_first is not None:
            self.fix_play_order = True
        else:
            self.fix_play_order = False

        self.reset()

    def reset(self):
        self.started = False
        self.matrix = np.zeros((3, 3))
        self.available_actions = list(range(9))
        if not self.fix_play_order:
            self.player_first = np.random.randint(2) % 2 == 0
        self.cpu = None

    def start(self):
        self.started = True

        if not self.player_first:
            self.cpu_action()

    def set_cpu(self, cpu):
        self.cpu = cpu

    def get_available_actions(self):
        return self.available_actions

    def get_game_settings(self):
        game_settings = {
            'player_first': self.player_first
        }

        return game_settings

    def step(self, action):
        '''
        returns: state after cpu action, following player action
        '''
        if not self.started:
            raise Exception('Game has not started.')
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
            self.matrix[0, 0] + self.matrix[1, 1] + self.matrix[2, 2],
            self.matrix[0, 2] + self.matrix[1, 1] + self.matrix[2, 0]
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

if __name__ == '__main__':
    game = TicTacToe()
    
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

    for i in range(n_episodes):
        game.reset()
        game.start()
        game_settings = game.get_game_settings()
        player_first = game_settings['player_first']

        state = 0
        while state == 0:
            available_actions = game.get_available_actions()
            action = np.random.choice(available_actions)
            state = game.step(action)

        if state == 1:
            if player_first:
                results[i] = 0
            else:
                results[i] = 3

        elif state == 2:
            if player_first:
                results[i] = 1
            else:
                results[i] = 4

        elif state == 3:
            if player_first:
                results[i] = 2
            else:
                results[i] = 5

        elif state == 9:
            results[i] = 9
            
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
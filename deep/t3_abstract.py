import sys
from collections import namedtuple, deque
from copy import deepcopy

import numpy as np
from tensorflow.python import keras as K

sys.path.append('.') # for debugging
from environment import T3Environment
from logger import Logger

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class Observer(T3Environment):
    @property
    def flatten_state(self):
        return self.get_state()

    def get_available_actions(self):
        return self.t3_game.get_available_actions()

    def step(self, action):
        return self.t3_game.step(action)

    def get_state(self, is_matrix=False):
        if is_matrix:
            return super().get_state(is_matrix)
        
        else:
            # not requested matrix, return flatten array instead of string
            state = super().get_state(True).reshape(9)
            state = state + 2
            state = state / 3
            return state

class Agent():
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.model = None
        self.use_prob_estimation = False
        self.initialized = False

    def save(self, model_path):
        self.model.save(model_path, overwrite=True, include_optimizer=False)

    @classmethod
    def load(cls, env: Observer, model_path, epsilon=1e-4):
        agent = cls(epsilon)
        agent.model = K.models.load_model(model_path)
        agent.initialized = True
        return agent

    def initialize(self, experiences):
        raise NotImplementedError()
        
    def estimate(self, state):
        raise NotImplementedError()
        
    def update(self, experiences, gamma):
        raise NotImplementedError()

    def select_action(self, state, actions):
        # Greedy-Epsilon method
        if np.random.rand() < self.epsilon or not self.initialized:
            # exploration
            action_index = np.random.randint(len(actions))
            action = actions[action_index]

        else:
            # exploitation
            estimates = self.estimate(state)
            available_estimates = estimates[actions]

            if self.use_prob_estimation:
                action = np.random.choice(actions, size=1, p=available_estimates)[0]

            else:
                action_index = np.argmax(available_estimates)
                action = actions[action_index]

        return action

    def play(self, env: Observer, episode_count=100):
        game = env.t3_game

        results = np.zeros(episode_count)
        for e in range(episode_count):
            game.reset()
            game.start()
            game_settings = game.get_game_settings()
            player_first = game_settings['player_first']
            
            state_type = 0
            while state_type == 0:
                available_actions = game.get_available_actions()
                state = env.get_state()

                action = self.select_action(state.reshape((-1, 9)), available_actions)
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

class Trainer():
    def __init__(self, buffer_size=1024, batch_size=32, gamma=0.9, report_interval=10, log_dir='log'):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.report_interval = report_interval
        
        self.logger = Logger(log_dir)

        self.experiences = deque(maxlen=buffer_size)
        self.has_trained = False # TODO rename
        self.training_count = 0
        self.reward_log = []

    @property
    def trainer_name(self):
        return self.__class__.__name__

    def reward_func(self, state_type):
        # default reward; game continuing
        # Winning faster is better, so default reward is minus value
        reward = 0.

        if state_type == 1:
            # win
            reward = 1.
        elif state_type == 2:
            # lose
            reward = -1.
        elif state_type == 9:
            # unexpected state
            raise Exception('invalid state')

        return reward

    def train_loop(self, env: Observer, agent: Agent, n_episodes=200, initial_count=-1, observe_interval=0):
        self.experiences = deque(maxlen=self.buffer_size)
        self.has_trained = False
        self.training_count = 0
        self.reward_log = []

        for i in range(n_episodes):
            env.reset()
            env.game_start()
            
            self.preprocess(i, agent)

            state_type = 0
            step_count = 0
            while state_type == 0:
                available_actions = env.get_available_actions()
                state = deepcopy(env.get_state())

                # Save Experience
                action = agent.select_action(state.reshape((-1, 9)), available_actions)
                state_type = env.step(action)
                next_state = deepcopy(env.get_state())
                reward = self.reward_func(state_type)

                e = Experience(state, action, reward, next_state, state_type!=0)
                self.experiences.append(e)

                if not self.has_trained and len(self.experiences) == self.buffer_size:
                    self.begin_train(i, agent)
                    self.has_trained = True

                self.step(i, step_count, agent, e)

                step_count += 1

            self.postprocess(i, step_count, agent)

            if not self.has_trained and initial_count > 0 and i >= initial_count:
                self.begin_train(i, agent)
                self.has_trained = True

            if self.has_trained:
                self.training_count += 1

    def preprocess(self, episode, agent):
        pass

    def begin_train(self, episode, agent):
        pass

    def step(self, episode, step_count, agent, experiences):
        pass

    def postprocess(self, episode, step_count, agent):
        pass

    def get_recent_experiences(self, count):
        recent_indices = range(len(self.experiences) - count, len(self.experiences))
        return [self.experiences[i] for i in recent_indices]

if __name__ == '__main__':
    env = Observer()
    agent = Agent(epsilon=1e-4)
    agent.play(env, 10000)
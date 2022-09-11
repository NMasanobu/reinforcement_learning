from collections import defaultdict
import math

import numpy as np
import matplotlib.pyplot as plt

class Agent():
    def __init__(self, env, epsilon):
        self.env = env
        # Use Epsilon-Greedy method
        self.epsilon = epsilon
        # Value table
        self.Q = {}
        # Counter corresponding Q table
        self.N = {}
        # record rewards for each episode
        self.reward_log = []

    def load(self, Q, N=None):
        self.Q = Q

        if N is not None:
            self.N = N

    def select_action(self, state, actions):
        action = None

        # Greedy-Epsilon method
        if np.random.rand() < self.epsilon:
            # exploration
            action_index = np.random.randint(len(actions))
            action = actions[action_index]

        else:
            # exploitation
            if state in self.Q and sum(self.Q[state]) != 0:
                # experienced state
                # G table includes unavailable actions
                values = np.array(self.Q[state])[actions]
                action_index = np.argmax(values)
                action = actions[action_index]

            else:
                # not experienced state
                action_index = np.random.randint(len(actions))
                action = actions[action_index]

        return action

    def clear_log(self):
        self.reward_log = []

    def log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self, interval=100):
        indices = list(range(0, len(self.reward_log), interval))
        means = []
        for i in indices:
            rewards = self.reward_log[i:(i + interval)]
            means.append(np.mean(rewards))
        plt.figure()
        plt.title('Reward History')
        plt.grid()
        plt.plot(means, label=f'Rewards for each {interval} episode.')
        plt.legend(loc='best')
        plt.show()

class MonteCarloAgent(Agent):
    def learn(self, episode_count=10000, gamma=0.9):
        self.clear_log()

        num_actions = list(range(9))
        self.Q = defaultdict(lambda: [0] * len(num_actions))
        self.N = defaultdict(lambda: [0] * len(num_actions))

        for e in range(episode_count):
            self.env.reset()
            self.env.game_start()

            # 1. Play until the end of game
            done = False
            experiences = []
            state = self.env.get_state()
            while not done:
                available_actions = self.env.get_available_actions_at(state)
                action = self.select_action(state, available_actions)
                
                # initialize defaultdict
                if state not in self.Q:
                    self.Q[state]
                    self.N[state]

                next_state, reward, done = self.env.step(action)
                experiences.append({'state': state, 'action': action, 'reward': reward})
                state = next_state

            self.log(reward)

            # 2. Evaluate each state and action
            for i, x in enumerate(experiences):
                s = x['state']
                a = x['action']

                # calculate discounted future reward of s
                G = 0
                for t, j in enumerate(range(i, len(experiences))):
                    # j: number of epochs from game start
                    # t: number of epochs from j
                    G += math.pow(gamma, t) * experiences[j]['reward']

                # update Q[s][a] with avarage of all episode (dinamically calculate an avarage)
                self.N[s][a] += 1
                alpha = 1 / self.N[s][a]
                self.Q[s][a] = alpha * (G - self.Q[s][a])

class QlearningAgent(Agent):
    def learn(self, episode_count=10000, gamma=0.9, lr=0.1):
        self.clear_log()

        num_actions = list(range(9))
        self.Q = defaultdict(lambda: [0] * len(num_actions))

        for e in range(episode_count):
            self.env.reset()
            self.env.game_start()

            # 1. Play until the end of game
            done = False
            state = self.env.get_state()
            while not done:
                available_actions = self.env.get_available_actions_at(state)
                action = self.select_action(state, available_actions)
                
                # initialize defaultdict
                if state not in self.Q:
                    self.Q[state]

                next_state, reward, done = self.env.step(action)
                
                gain = reward + gamma * max(self.Q[state])
                estimated = self.Q[state][action]
                self.Q[state][action] = estimated + lr * (gain - estimated)

                state = next_state

            self.log(reward)

class SARSAAgent(Agent):
    def learn(self, episode_count=10000, gamma=0.9, lr=0.1):
        self.clear_log()

        num_actions = list(range(9))
        self.Q = defaultdict(lambda: [0] * len(num_actions))

        for e in range(episode_count):
            self.env.reset()
            self.env.game_start()

            # 1. Play until the end of game
            done = False
            state = self.env.get_state()
            while not done:
                available_actions = self.env.get_available_actions_at(state)
                action = self.select_action(state, available_actions)
                
                # initialize defaultdict
                if state not in self.Q:
                    self.Q[state]

                next_state, reward, done = self.env.step(action)
                # On-Policy
                next_available_actions = self.env.get_available_actions_at(next_state)
                next_action = self.select_action(next_state, next_available_actions)

                gain = reward + gamma * self.Q[state][next_action]
                estimated = self.Q[state][action]
                self.Q[state][action] = estimated + lr * (gain - estimated)

                state = next_state

            self.log(reward)

if __name__ == '__main__':
    #agent = MonteCarloAgent(0.1)
    hoge = defaultdict(lambda: [0] * len(fuga))

    for i in range(0, 4):
        fuga = ''.join(['x'] * i)
        print(hoge[i])
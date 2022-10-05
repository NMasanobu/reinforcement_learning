import random
import argparse

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor

from t3_abstract import Agent, Trainer, Observer

class ValueFunctionAgent(Agent):
    def save(self, model_path):
        '''
        Original method
        '''
        joblib.dump(self.model, model_path)

    # extends
    @classmethod
    def load(cls, env: Observer, model_path, epsilon=1e-4):
        agent = cls(epsilon)
        agent.model = joblib.load(model_path)
        agent.initialized = True
        return agent

    def initialize(self, experiences):
        self.model = MLPRegressor(hidden_layer_sizes=(16, 32, 16), max_iter=1)

        # Avoid predicting before fitting
        self.update([experiences[0]], gamma=0)
        self.initialized = True
        
    def estimate(self, state):
        '''
        model prediction for a single state
        used in playing
        '''
        return self.model.predict(state)[0]

    def update(self, experiences, gamma):
        states = np.vstack([e.state for e in experiences])
        next_states = np.vstack([e.next_state for e in experiences])

        estimateds = self._predict(states)
        futures = self._predict(next_states)

        for i, e in enumerate(experiences):
            reward = e.reward
            if not e.done:
                reward += gamma * np.max(futures[i])

            estimateds[i][e.action] = reward

        estimateds = np.array(estimateds)
        self.model.partial_fit(states, estimateds)

    def _predict(self, states):
        '''
        model prediction for multi states
        used in updating
        '''
        if self.initialized:
            predicteds = self.model.predict(states)
        else:
            # initialized by uniform
            size = 9 * len(states)
            predicteds = np.random.uniform(size=size)
            predicteds = predicteds.reshape((-1, 9))

        return predicteds


class ValueFanctionTrainer(Trainer):
    def train(self, env: Observer, n_episodes=100000, epsilon=0.1, initial_count=-1):
        agent = ValueFunctionAgent(epsilon)
        self.train_loop(env, agent, n_episodes, initial_count)
        
        return agent

    # extends
    def begin_train(self, episode, agent: ValueFunctionAgent):
        agent.initialize(self.experiences)

    def step(self, episode, step_count, agent: ValueFunctionAgent, experiences):
        if self.has_trained:
            batch = random.sample(self.experiences, self.batch_size)
            agent.update(batch, self.gamma)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true')

    args = parser.parse_args()
    is_play_mode = args.play

    env = Observer()
    trainer = ValueFanctionTrainer()
    model_path = 'output/vf_agent.pkl'

    if is_play_mode:
        agent = ValueFunctionAgent.load(env, model_path, epsilon=0)
        agent.play(env, episode_count=10000)

    else:
        trained_agent = trainer.train(env, epsilon=1.)
        trained_agent.save(model_path)
        trained_agent.epsilon = 0
        trained_agent.play(env, episode_count=10000)

import random
import argparse

import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Optimizer

from t3_abstract import Agent, Trainer, Observer, Experience

tf.compat.v1.disable_eager_execution()

class PolicyGradientAgent(Agent):
    def __init__(self, epsilon):
        super().__init__(epsilon)
        self.estimate_probs = True
        self._updater = None

    # extends
    @classmethod
    def load(cls, env: Observer, model_path, epsilon=1e-4):
        agent = cls(epsilon)
        agent.model = joblib.load(model_path)
        agent.initialized = True
        return agent

    def initialize(self, experiences, optimizer):
        states = np.vstack([e.state for e in experiences])
        feature_size = states.shape[1]

        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(feature_size,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(9, activation='softmax')
        ])
        self.set_updater(optimizer)
        self.initialized = True

    def set_updater(self, optimizer: Optimizer):
        # array of action id of each experience
        actions = tf.compat.v1.placeholder(shape=(None), dtype='int32')
        rewards = tf.compat.v1.placeholder(shape=(None), dtype='float32')
        
        one_hot_actions = tf.one_hot(actions, depth=9, axis=1)
        action_probs = self.model.output
        selected_action_probs = tf.reduce_sum(one_hot_actions * action_probs, axis=1)

        clipped = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)
        loss = -tf.math.log(clipped) * rewards
        loss = tf.reduce_mean(loss)

        updates = optimizer.get_updates(loss=loss, params=self.model.trainable_weights)

        self._updater = K.function(inputs=[self.model.input, actions, rewards], outputs=[loss], updates=updates)
        
    def estimate(self, state):
        '''
        model prediction for a single state
        used in playing
        '''
        return self.model.predict(state)[0]

    def update(self, states, actions, rewards):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        self._updater([states, actions, rewards])

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


class PolicyGradientTrainer(Trainer):
    def train(self, env: Observer, n_episodes=100000, epsilon=0.1, initial_count=-1):
        agent = PolicyGradientAgent(epsilon)
        self.train_loop(env, agent, n_episodes, initial_count)
        
        return agent

    # extends
    def preprocess(self, episode, agent):
        if agent.initialized:
            self.experiences = []

    def make_batch(self, policy_experiences):
        length = min(self.batch_size, len(policy_experiences))
        batch = random.sample(policy_experiences, length)

        states = np.vstack([e.state for e in batch])
        actions = [e.action for e in batch]
        rewards = [e.reward for e in batch]

        scaler = StandardScaler()
        rewards = np.array(rewards).reshape((-1, 1))
        rewards = scaler.fit_transform(rewards).flatten()

        return states, actions, rewards

    def postprocess(self, episode, step_count, agent: PolicyGradientAgent):
        rewards = [e.reward for e in self.get_recent_experiences(step_count)]

        if not agent.initialized:
            if len(self.experiences) == self.buffer_size:
                optimizer = Adam(learning_rate=0.01)
                agent.initialize(self.experiences, optimizer)
                self.training = True

        else:
            policy_experiences = []
            for t, e in enumerate(self.experiences):
                s, a, r, n_s, d = e
                disounted_r = [_r * (self.gamma ** i) for i, _r in enumerate(rewards[t:])]
                disounted_r = sum(disounted_r)
                d_e = Experience(s, a, disounted_r, n_s, d)
                policy_experiences.append(d_e)

            agent.update(*self.make_batch(policy_experiences))

    def reward_func(self, state_type):
        return super().reward_func(state_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true')

    args = parser.parse_args()
    is_play_mode = args.play

    env = Observer()
    trainer = PolicyGradientTrainer()
    model_path = 'output/pg_agent'

    if is_play_mode:
        agent = PolicyGradientAgent.load(env, model_path, epsilon=0)
        agent.play(env, episode_count=10000)

    else:
        trained_agent = trainer.train(env, epsilon=0.1)
        trained_agent.save(model_path)
        trained_agent.epsilon = 0
        trained_agent.play(env, episode_count=10000)

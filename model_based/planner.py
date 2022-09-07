from collections import defaultdict

class Planner():
    def __init__(self, env):
        self.env = env
        self.log = []

    def reset(self):
        self.env.reset()
        self.log = []

    def plan(self, gamma=0.9, thresh=1e-4):
        raise NotImplementedError()

    def transitions_at(self, state, action):
        transition_dict = self.env.transit_func(state, action)
        for next_state, prob in transition_dict.items():
            reward = self.env.reward_func(next_state)

            yield prob, next_state, reward

class ValueIterationPlanner(Planner):
    def __init__(self, env):
        super().__init__(env)

    def plan(self, gamma=0.9, thresh=1e-4, max_iter=100000):
        self.reset()

        V = {}

        for s in self.env.states:
            V[s] = 0

        counter = 0
        while counter < max_iter:
            delta = 0

            for s in V:
                state_type = self.env.check_state(s)
                if state_type != 0:
                    # game end or invalid state
                    continue
                
                expected_rewards = []
                for a in self.env.actions:
                    if not self.env.is_valid_action(s, a):
                        # cannot select already filled cell
                        continue

                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])

                    expected_rewards.append(r)

                max_reward = max(expected_rewards)
                # use max difference between expected reward and actual reward
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward

            if delta < thresh:
                break

            counter += 1

        else:
            print('exceed max_iter')

        return V

class PolicyIterationPlanner(Planner):
    def __init__(self, env):
        super().__init__(env)
        self.policy = {}

    def reset(self):
        super().reset()
        self.policy = {}

        for s in self.env.states:
            self.policy[s] = defaultdict(float)
            
            available_actions = self.env.get_available_actions_at(s)
            n_actions = len(available_actions)

            for a in available_actions:
                self.policy[s][a] = 1 / n_actions

    def estimate_value(self, gamma, thresh, max_iter=100000):
        '''
        estimate value with policy
        '''
        V = {}

        for s in self.env.states:
            V[s] = 0

        counter = 0
        while counter < max_iter:
            delta = 0

            for s in V:
                state_type = self.env.check_state(s)
                if state_type != 0:
                    # game end or invalid state
                    continue
                
                expected_rewards = []
                for a in self.env.actions:
                    if not self.env.is_valid_action(s, a):
                        # cannot select already filled cell
                        continue

                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])

                    expected_rewards.append(r)

                reward = sum(expected_rewards)
                
                delta = max(delta, abs(reward - V[s]))
                V[s] = reward

            if delta < thresh:
                break

            counter += 1

        else:
            print('exceed max_iter')

        return V

    def plan(self, gamma=0.9, thresh=1e-4, max_iter=100000):
        self.reset()

        counter = 0
        while counter < max_iter:
            updated = False

            # estimate expected rewards under current policy
            V = self.estimate_value(gamma, thresh)

            for s in V:
                state_type = self.env.check_state(s)
                if state_type != 0:
                    # game end or invalid state
                    continue
                
                # get an action following to the current policy
                policy_action = self.take_max_action(self.policy[s])

                # compare with other actions
                action_rewards = {}
                for a in self.env.actions:
                    if not self.env.is_valid_action(s, a):
                        # cannot select already filled cell
                        continue

                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])

                    action_rewards[a] = r

                best_action = self.take_max_action(action_rewards)

                if policy_action != best_action:
                    updated =  True

                # update policy
                for a in self.policy[s]:
                    if a == best_action:
                        self.policy[s][a] = 1.
                    else:
                        self.policy[s][a] = 0.

            if not updated:
                break

            counter += 1

        else:
            print('exceed max_iter')

        return V, self.policy

    def take_max_action(self, action_value_dict):
        return max(action_value_dict, key=action_value_dict.get)

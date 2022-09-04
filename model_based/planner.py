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

        actions = self.env.actions
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
                for a in actions:
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
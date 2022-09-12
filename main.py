from copy import deepcopy
import json
import argparse

import numpy as np

from game.tic_tac_toe import TicTacToe
from environment import T3Environment
from model_based.planner import ValueIterationPlanner, PolicyIterationPlanner
from model_free.t3_agent import MonteCarloAgent, QlearningAgent, SARSAAgent, ActorCritic

TARGET_LIST = ['mb_vi', 'mb_pi', 'mf_mc', 'mf_ql', 'mf_sarsa', 'mf_ac']

def train(target):
    env = T3Environment()

    if target == 'mb_vi':
        planner = ValueIterationPlanner(env)
        V = planner.plan()

        with open('output/model_based_vi.json', mode='w') as f:
            json.dump(V, f)

    elif target == 'mb_pi':
        planner = PolicyIterationPlanner(env)
        V, policy = planner.plan()

        with open('output/model_based_pi_value.json', mode='w') as f:
            json.dump(V, f)

        with open('output/model_based_pi_policy.json', mode='w') as f:
            json.dump(policy, f)

    elif target == 'mf_mc':
        # only exploration
        agent = MonteCarloAgent(env, epsilon=1.)
        agent.learn(episode_count=1000000)

        # if less than 4520, it means that episode_count is too small
        print(len(agent.Q))
        
        with open('output/model_free_mc_Q.json', mode='w') as f:
            json.dump(agent.Q, f)

    elif target == 'mf_ql':
        # only exploration
        agent = QlearningAgent(env, epsilon=1.)
        agent.learn(episode_count=1000000)

        # if less than 4520, it means that episode_count is too small
        print(len(agent.Q))
        
        with open('output/model_free_ql_Q.json', mode='w') as f:
            json.dump(agent.Q, f)

    elif target == 'mf_sarsa':
        agent = SARSAAgent(env, epsilon=.2)
        agent.learn(episode_count=5000000)

        # if less than 4520, it means that episode_count is too small
        print(len(agent.Q))
        
        with open('output/model_free_sarsa_Q.json', mode='w') as f:
            json.dump(agent.Q, f)
            
    elif target == 'mf_ac':
        agent = ActorCritic(env)
        agent.learn(episode_count=1000000)

        # if less than 4520, it means that episode_count is too small
        print(len(agent.actor.Q))
        
        with open('output/model_free_ac_Q.json', mode='w') as f:
            Q = {}
            for s, action_probs in agent.actor.Q.items():
                Q[s] = action_probs.tolist()
            json.dump(Q, f)
        
        with open('output/model_free_ac_value.json', mode='w') as f:
            json.dump(agent.critic.V, f)


def play(target):
    game = TicTacToe()
    env = T3Environment()

    # Use one of them
    V = None # Value of each state
    P = None # Policy
    agent = None

    if target == 'mb_vi':
        with open('output/model_based_vi.json') as f:
            V = json.load(f)
    
    elif target == 'mb_pi':
        with open('output/model_based_pi_policy.json') as f:
            P = json.load(f)

    elif target == 'mf_mc':
        with open('output/model_free_mc_Q.json') as f:
            Q = json.load(f)
        agent = MonteCarloAgent(env, epsilon=0.)
        agent.load(Q)

    elif target == 'mf_ql':
        with open('output/model_free_ql_Q.json') as f:
            Q = json.load(f)
        agent = QlearningAgent(env, epsilon=0.)
        agent.load(Q)

    elif target == 'mf_sarsa':
        with open('output/model_free_sarsa_Q.json') as f:
            Q = json.load(f)
        agent = SARSAAgent(env, epsilon=0.)
        agent.load(Q)

    elif target == 'mf_sarsa':
        with open('output/model_free_sarsa_Q.json') as f:
            Q = json.load(f)
        agent = ActorCritic(env, epsilon=0.)
        agent.load(Q)

    elif target == 'mf_ac':
        with open('output/model_free_ac_Q.json') as f:
            Q = json.load(f)
        agent = ActorCritic(env).actor
        agent.load(Q)

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
            state_str = env.matrix2state(game.matrix)

            action = select_action(available_actions, state_str, V, P, agent)
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

def select_action(available_actions, state_str, V=None, P=None, agent=None):
    if V is not None:
        # value base
        value_list = [] # order coresponds that of avaliable_actions
        for a in available_actions:
            next_state_list = []
            state_str_list = list(state_str)

            # apply action
            state_str_list[a] = '1'

            if not '0' in state_str_list:
                # all cells are filled
                next_state_list.append(''.join(state_str_list))

            else:
                # posible states after cpu action
                for i in range(9):
                    if state_str_list[i] == '0':
                        state_list_copy = deepcopy(state_str_list)
                        state_list_copy[i] = '2'
                        next_state_list.append(''.join(state_list_copy))

            # maximize expected value
            value_list.append(np.mean([V[s] for s in next_state_list]))

        # maximize expected value
        action = available_actions[np.argmax(value_list)]

    elif P is not None:
        # In policy P, actions are defined as string
        available_actions_str = [str(a) for a in available_actions]

        # Policy P has probability of available actions
        action = int(max(available_actions_str, key=P[state_str].get))

    elif agent is not None:
        action = agent.select_action(state_str, available_actions)

    return action

def validate_actions_at(state_str='002010000', player_First=True):
    game = TicTacToe()
    env = T3Environment()

    for target in TARGET_LIST:
        # Use one of them
        V = None # Value of each state
        agent = None

        if target == 'mb_vi':
            with open('output/model_based_vi.json') as f:
                V = json.load(f)
        
        elif target == 'mb_pi':
            with open('output/model_based_pi_value.json') as f:
                V = json.load(f)

        elif target == 'mf_mc':
            from model_free.t3_agent import MonteCarloAgent
            
            with open('output/model_free_mc_Q.json') as f:
                Q = json.load(f)
            agent = MonteCarloAgent(env, epsilon=0.)
            agent.load(Q)

        elif target == 'mf_ql':
            from model_free.t3_agent import QlearningAgent
            
            with open('output/model_free_ql_Q.json') as f:
                Q = json.load(f)
            agent = QlearningAgent(env, epsilon=0.)
            agent.load(Q)

        available_actions = env.get_available_actions_at(state_str)
        value_list = []

        if V is not None:
            # value base
            value_list = [] # order coresponds that of avaliable_actions
            for a in available_actions:
                next_state_list = []
                state_str_list = list(state_str)

                # apply action
                state_str_list[a] = '1'

                if not '0' in state_str_list:
                    # all cells are filled
                    next_state_list.append(''.join(state_str_list))

                else:
                    # posible states after cpu action
                    for i in range(9):
                        if state_str_list[i] == '0':
                            state_list_copy = deepcopy(state_str_list)
                            state_list_copy[i] = '2'
                            next_state_list.append(''.join(state_list_copy))

                # maximize expected value
                value_list.append(np.mean([V[s] for s in next_state_list]))

        elif agent is not None:
            value_list = agent.Q[state_str]

        print(value_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train',
                        help='Train target. If not given, train method is not executed.',
                        choices=TARGET_LIST)
    parser.add_argument('-p', '--play',
                        help='Play target. If not given, play method is not executed.',
                        choices=TARGET_LIST)
    parser.add_argument('-d', '--demo',
                        action='store_true',
                        help='Validate actions for passed state.')

    args = parser.parse_args()
    train_target = args.train
    play_target = args.play
    do_demo = args.demo

    if not (play_target or train_target or do_demo):
        print('At least one of optional targets -t, -p or -d is required.')
        parser.print_help()
    
    if train_target:
        train(train_target)

    if play_target:
        play(play_target)

    if do_demo:
        validate_actions_at()
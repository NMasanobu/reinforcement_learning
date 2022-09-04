from copy import deepcopy

from model_based.environment import T3Environment
from model_based.planner import ValueIterationPlanner

def main():
    env = T3Environment()
    planner = ValueIterationPlanner(env)
    V = planner.plan()

    state = '000000000'
    state_list = list(state)
    for i in range(9):
        state_list_copy = deepcopy(state_list)
        state_list_copy[i] = '1'
        state_ = ''.join(state_list_copy)
        print(f'action {i} value {V[state_]}')

    print()

    print(V['122010120'])
    print(V['122010102'])

if __name__ == '__main__':
    main()
from copy import deepcopy
import json

from model_based.environment import T3Environment
from model_based.planner import ValueIterationPlanner

def main():
    env = T3Environment()
    planner = ValueIterationPlanner(env)
    V = planner.plan()

    with open('output/model_based_vi.json', mode='w') as f:
        json.dump(V, f)

if __name__ == '__main__':
    main()
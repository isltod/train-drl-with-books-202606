import warnings

warnings.filterwarnings("ignore")

import gymnasium as gym, gym_walk, gym_aima
import numpy as np
from pprint import pprint
from tqdm import tqdm_notebook as tqdm

from itertools import cycle

import random
import itertools
import tabulate

np.set_printoptions(suppress=True)
random.seed(123)
np.random.seed(123)


def print_policy(pi, P, action_symbols=("<", "v", ">", "^"), n_cols=4, title="정책:"):
    print(title)
    arrs = {k: v for k, v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0:
            print("|")


def print_state_value_function(V, P, n_cols=4, prec=3, title="상태-가치 함수:"):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), "{}".format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0:
            print("|")


def print_action_value_function(
    Q, optimal_Q=None, action_symbols=("<", ">"), prec=3, title="행동-가치 함수:"
):
    vf_types = ("",) if optimal_Q is None else ("", "*", "err")
    headers = [
        "s",
    ] + [" ".join(i) for i in list(itertools.product(vf_types, action_symbols))]
    print(title)
    states = np.arange(len(Q))[..., np.newaxis]
    arr = np.hstack((states, np.round(Q, prec)))
    if not (optimal_Q is None):
        arr = np.hstack((arr, np.round(optimal_Q, prec), np.round(optimal_Q - Q, prec)))
    print(tabulate(arr, headers, tablefmt="fancy_grid"))


def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123)
    np.random.seed(123)
    env.reset(seed=123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            print(type(state))

            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results) / len(results)


def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123)
    np.random.seed(123)
    env.reset(seed=123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)


env = gym.make("SlipperyWalkFive-v0")
P = env.unwrapped.P
init_state = env.reset()
goal_state = 6

LEFT, RIGHT = range(2)


pi = lambda s: {0: LEFT, 1: LEFT, 2: LEFT, 3: LEFT, 4: LEFT, 5: LEFT, 6: LEFT}[s]
print_policy(pi, P, action_symbols=("<", ">"), n_cols=7)
print(
    "Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.".format(
        probability_success(env, pi, goal_state=goal_state) * 100, mean_return(env, pi)
    )
)

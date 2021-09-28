from functools import partial
import gym
import pybullet_envs
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import time
import numpy as np

if __name__ == "__main__":
    env_name = sys.argv[1]
    
    n_envs = 10
    n_steps = 10000
    env_fn = partial(gym.make, env_name)

    env = DummyVecEnv([env_fn for _ in range(n_envs)])
    env.reset()

    act = np.zeros((n_envs, env.action_space.shape[0]))    
    start = time.time()
    for i in range(n_steps):
        env.step(act)
    print(f"{env_name} Dummy: {n_steps * n_envs / (time.time() - start)}")
    env.close()
    

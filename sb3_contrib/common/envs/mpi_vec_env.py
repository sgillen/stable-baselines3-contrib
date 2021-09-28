
import numpy as np
from mpi4py import MPI# This will call MPI_Init()
from stable_baselines3.common.vec_env import VecEnv
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
import gym
import time

STEP_CMD_ID = 1
RESET_CMD_ID = 2
CLOSE_CMD_ID = 3

def worker_fn(comm, env_fn):
    env = env_fn()

    obs_data = np.empty(env.observation_space.shape, env.observation_space.dtype)
    act_data = np.empty(env.action_space.shape, env.action_space.dtype)
    rew_data = np.empty(1, dtype=np.float64)
    done_data = np.empty(1, dtype=bool)

    while True:
        cmd = comm.bcast(None, root=0) # TODO, would love for this to be fast too ..

        if cmd == STEP_CMD_ID:
            comm.Scatter(None, act_data, root=0)
            obs_data[:], rew_data[:], done_data[:], info = env.step(act_data)
            comm.Gather(obs_data, None, root=0)
            comm.Gather(rew_data, None, root=0)
            comm.Gather(done_data, None, root=0)
            comm.gather(info, root=0)

        if cmd == RESET_CMD_ID:
            obs_data[:] = env.reset()
            comm.Gather(obs_data, None, root=0)

        if cmd == CLOSE_CMD_ID:
            env.close()
            return


class MpiVecEnv(VecEnv):
    def __init__(self, env_fns):

        self.comm = MPI.COMM_WORLD
        rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_envs = len(env_fns)

        assert self.size-1 == self.num_envs

        if rank == 0:
            pass
        else:
            worker_fn(self.comm, env_fns[rank-1])
            exit()

        env = env_fns[0]() # TODO get this from workers
        observation_space = env.observation_space
        action_space = env.action_space

        self.act_buff = np.empty([self.size, *env.action_space.shape], dtype=env.action_space.dtype)
        self.obs_buff = np.empty([self.size, *env.observation_space.shape], dtype=env.observation_space.dtype)
        self.rew_buff = np.empty([self.size], dtype=np.float64) # is this ... always true?
        self.done_buff = np.empty([self.size], dtype=bool)

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        pass

    def step_async(self, actions):
        self.comm.bcast(STEP_CMD_ID , root=0)
        self.act_buff[1:, :] = actions  # hack
        self.comm.Scatter(self.act_buff, MPI.IN_PLACE, root=0)

    def step_wait(self):
        self.comm.Gather(MPI.IN_PLACE, self.obs_buff, root=0)
        self.comm.Gather(MPI.IN_PLACE, self.rew_buff, root=0)
        self.comm.Gather(MPI.IN_PLACE, self.done_buff, root=0)
        infos = self.comm.gather(MPI.IN_PLACE, root=0)

        return self.obs_buff[1:,:], self.rew_buff[1:], self.done_buff[1:], infos[1:]

    def reset(self):
        self.comm.bcast(RESET_CMD_ID, root=0)
        self.comm.Gather(MPI.IN_PLACE, self.obs_buff)
        return self.obs_buff[1:,:]

    # TODO why does subproc vec env not have to implement these?

    def close(self):
        self.comm.bcast(CLOSE_CMD_ID, root=0)
#        self.comm.Disconnect()
        # TODO make sure everyone is done
        return

    def get_attr(self, attr_name: str, indices = None):
        pass

    def set_attr(self, attr_name: str, value: Any, indices = None) -> None:
        pass

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices = None) -> List[bool]:
        pass

    def env_method(self, method_name: str, *method_args, indices = None, **method_kwargs) -> List[Any]:
        pass

    def seed(self, seed):
        pass




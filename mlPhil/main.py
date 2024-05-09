import os
import torch.multiprocessing as mp
from parallel_env import ParallelEnv

os.environ['OMP_NUM_THREADS'] = '1'


if __name__ == '__main__':
    mp.set_start_method('spawn')
    env_id = 'MiniGrid-MultiRoom-N4-S5-v0'
    n_threads = 12
    n_actions = 7
    input_shape = [147]
    env = ParallelEnv(env_id=env_id, n_threads=n_threads,
                      n_actions=n_actions, input_shape=input_shape, icm=False)
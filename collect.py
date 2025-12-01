import os
from datetime import datetime
import yaml
import multiprocessing

from env import make_env
from algorithm import ALGORITHM, HistoryLoggerCallback
import h5py
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv
import metaworld
from gymnasium.wrappers import TimeLimit

from utils import get_config, get_traj_file_name



def worker(config, env_cls, task_instance, traj_dir, env_idx, history, file_name):
    
    env = DummyVecEnv([make_env(config, env_cls, task_instance)] * config['n_stream'])
    
    alg_name = config['alg']
    seed = config['alg_seed'] + env_idx

    config['device'] = 'cpu'

    alg = ALGORITHM[alg_name](config, env, seed, traj_dir)
    callback = HistoryLoggerCallback(config['env'], env_idx, history)
    log_name = f'{file_name}_{env_idx}'
    
    alg.learn(total_timesteps=config['total_source_timesteps'],
              callback=callback,
              log_interval=1,
              tb_log_name=log_name,
              reset_num_timesteps=True,
              progress_bar=False)
    env.close()



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    config = get_config("config/env/metaworld.yaml")
    config.update(get_config("config/algorithm/ppo_metaworld.yaml"))

    if not os.path.exists("datasets"):
        os.makedirs("datasets", exist_ok=True)
        
    task = config['task']
    
    ml1 = metaworld.ML1(env_name=task, seed=config['mw_seed'])
        
    file_name = get_traj_file_name(config)
    
    # Collect train task histories
    name, env_cls = list(ml1.train_classes.items())[0]
    task_instances = ml1.train_tasks
    path = f'datasets/{task}/'
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    start_time = datetime.now()
    print(f'Training started at {start_time}')

    with h5py.File(os.path.join(path, f'{file_name}.hdf5'), 'a') as f:
        start_idx = 0
        
        while f'{start_idx}' in f.keys():
            start_idx += 1
            
        with multiprocessing.Manager() as manager:
            
            while start_idx < len(task_instances):
                history = manager.dict()

                instances = task_instances[start_idx:start_idx+config['n_process']]
                
                with multiprocessing.Pool(processes=config['n_process']) as pool:
                    pool.starmap(worker, [(config, env_cls, task_instance, path, start_idx+i, history, file_name) for i, task_instance in enumerate(instances)])

                # Save the history dictionary
                for i in range(start_idx, start_idx+len(instances)):
                    env_group = f.create_group(f'{i}')
                    for key, value in history[i].items():
                        env_group.create_dataset(key, data=value)
            
                start_idx += len(instances)

    end_time = datetime.now()
    print()
    print(f'Training ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
    
    # Collect test task histories
    name, env_cls = list(ml1.test_classes.items())[0]
    task_instances = ml1.test_tasks[:10]
    path = f'datasets/{task}/test/'
        
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
    start_time = datetime.now()
    print(f'Collecting test task histories started at {start_time}')
    
    print()
    
    with h5py.File(f'{path}/{file_name}.hdf5', 'a') as f:
        start_idx = 0
        
        while f'{start_idx}' in f.keys():
            start_idx += 1
            
        with multiprocessing.Manager() as manager:
            
            while start_idx < len(task_instances):
                history = manager.dict()

                instances = task_instances[start_idx:start_idx+config['n_process']]
                
                with multiprocessing.Pool(processes=config['n_process']) as pool:
                    pool.starmap(worker, [(config, env_cls, task_instance, path, start_idx+i, history, file_name) for i, task_instance in enumerate(instances)])

                # Save the history dictionary
                for i in range(start_idx, start_idx+len(instances)):
                    env_group = f.create_group(f'{i}')
                    for key, value in history[i].items():
                        env_group.create_dataset(key, data=value)
            
                start_idx += len(instances)

    end_time = datetime.now()
    print()
    print(f'Collecting test task histories ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')

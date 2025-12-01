from datetime import datetime

from glob import glob

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import os.path as path

from env import make_env
from model import MODEL
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import metaworld
from gymnasium.wrappers import TimeLimit

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
seed = 0
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    ckpt_dir = './runs/AD-metaworld-pick-place-v3'
    ckpt_paths = sorted(glob(path.join(ckpt_dir, 'ckpt-*.pt')))

    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path)
        print(f'Checkpoint loaded from {ckpt_path}')
        config = ckpt['config']
    else:
        raise ValueError('No checkpoint found.')
    
    model_name = config['model']
    model = MODEL[model_name](config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    env_name = config['env']
    
    if env_name == 'metaworld':
        ml1 = metaworld.ML1(env_name=config['task'], seed=config['mw_seed'])
        
        test_envs = []
        
        for task_name, env_cls in ml1.test_classes.items():
            task_instances = [task for task in ml1.test_tasks if task.env_name == task_name]
            for i in range(50):
                test_envs.append(make_env(config, env_cls, task_instances[i]))
        
        envs = SubprocVecEnv(test_envs)
        
        if config['task'] == 'reach-v2':
            eval_episodes = 100
        elif config['task'] == 'push-v2':
            eval_episodes = 300
        elif config['task'] == 'pick-place-v2' or config['task'] == 'peg-insert-side-v2':
            eval_episodes = 2000
        else:
            eval_episodes = 200
    else:
        raise NotImplementedError(f'Environment {env_name} not supported')
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    start_time = datetime.now()
    print(f'Starting at {start_time}')
    
    # For CompressedAD: automatic hierarchical compression during evaluation
    # When sequence length exceeds max, encoder compresses context into latents
    with torch.no_grad():
        output = model.evaluate_in_context(vec_env=envs, eval_timesteps=eval_episodes * config['horizon'])
        result_path = path.join(ckpt_dir, 'eval_result')
    
    end_time = datetime.now()
    print()
    print(f'Ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')

    envs.close()

    reward_episode = output['reward_episode']
    success = output['success']
    
    with open(f'{result_path}_reward.npy', 'wb') as f:
        np.save(f, reward_episode)
    with open(f'{result_path}_success.npy', 'wb') as f:
        np.save(f, success)

    print("Mean reward per environment:", reward_episode.mean(axis=1))
    print("Overall mean reward: ", reward_episode.mean())
    print("Std deviation: ", reward_episode.std())
    print("Success rate: ", success.mean())
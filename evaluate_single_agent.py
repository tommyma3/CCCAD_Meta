"""
Evaluate a trained single PPO agent on MetaWorld.
Loads a saved model and runs evaluation episodes.
"""

import os
import argparse
import numpy as np
from datetime import datetime

import metaworld
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit

from utils import get_config


def make_env(config, env_cls, task):
    """Create a single MetaWorld environment."""
    def _init():
        env = env_cls()
        env.set_task(task)
        return TimeLimit(env, max_episode_steps=config['horizon'])
    return _init


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO agent on MetaWorld')
    parser.add_argument('--model-path', '-m', type=str, required=True,
                        help='Path to the trained model (.zip file)')
    parser.add_argument('--task', '-t', type=str, default='pick-place-v3',
                        help='MetaWorld task name (default: pick-place-v3)')
    parser.add_argument('--episodes', '-e', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--task-index', type=int, default=0,
                        help='Task instance index from train_tasks (default: 0)')
    parser.add_argument('--render', '-r', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--deterministic', '-d', action='store_true', default=True,
                        help='Use deterministic actions (default: True)')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    
    # Load configs
    env_config = get_config('./config/env/metaworld.yaml')
    env_config['task'] = args.task
    
    print("="*50)
    print("EVALUATION CONFIGURATION")
    print("="*50)
    print(f"Model Path: {args.model_path}")
    print(f"Task: {args.task}")
    print(f"Task Instance Index: {args.task_index}")
    print(f"Episodes: {args.episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Render: {args.render}")
    print("="*50)
    print()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Initialize MetaWorld environment
    print("Initializing MetaWorld environment...")
    ml1 = metaworld.ML1(env_name=args.task, seed=env_config['mw_seed'])
    
    # Get environment class and task instance
    env_name, env_cls = list(ml1.train_classes.items())[0]
    task_instance = ml1.train_tasks[args.task_index]
    
    print(f"Environment: {env_name}")
    print(f"Task Instance: {args.task_index}")
    print()
    
    # Create environment
    env = DummyVecEnv([make_env(env_config, env_cls, task_instance)])
    
    # Load the trained model
    print("Loading trained model...")
    model = PPO.load(args.model_path, env=env)
    print("Model loaded successfully!")
    print()
    
    # Evaluate the agent
    start_time = datetime.now()
    print(f"Evaluation started at {start_time}")
    print()
    
    eval_rewards = []
    eval_successes = []
    eval_lengths = []
    
    obs = env.reset()
    for episode in range(args.episodes):
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
            if args.render:
                env.render()
            
            if done[0]:
                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
                if 'success' in info[0]:
                    eval_successes.append(info[0]['success'])
                
                # Print progress
                if (episode + 1) % 10 == 0:
                    mean_reward = np.mean(eval_rewards)
                    mean_success = np.mean(eval_successes) if eval_successes else 0.0
                    print(f"Episode {episode + 1}/{args.episodes}: "
                          f"Mean Reward: {mean_reward:.2f}, "
                          f"Mean Success: {mean_success:.2%}")
                
                obs = env.reset()
                break
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    print()
    print(f"Evaluation ended at {end_time}")
    print(f"Elapsed time: {elapsed}")
    print()
    
    # Print results
    print("="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Episodes: {args.episodes}")
    print(f"Mean Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Min Reward: {np.min(eval_rewards):.2f}")
    print(f"Max Reward: {np.max(eval_rewards):.2f}")
    print(f"Median Reward: {np.median(eval_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(eval_lengths):.2f} ± {np.std(eval_lengths):.2f}")
    if eval_successes:
        print(f"Success Rate: {np.mean(eval_successes):.2%} ({np.sum(eval_successes)}/{len(eval_successes)})")
    print("="*50)
    
    # Save results
    results_dir = os.path.dirname(args.model_path)
    results_path = os.path.join(results_dir, f'eval_task{args.task_index}_episodes{args.episodes}.npz')
    np.savez(results_path,
             rewards=np.array(eval_rewards),
             successes=np.array(eval_successes) if eval_successes else np.array([]),
             lengths=np.array(eval_lengths))
    print(f"\nResults saved to {results_path}")
    
    env.close()
    print("Done!")


if __name__ == '__main__':
    main()

"""
Train a single PPO agent on a single MetaWorld environment.
This script helps you visualize how the source algorithm (PPO) behaves
before applying Algorithm Distillation.
"""

import os
from datetime import datetime
import yaml
import argparse

import metaworld
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
import numpy as np

from utils import get_config


class MetricsCallback(BaseCallback):
    """
    Callback for logging training metrics.
    """
    def __init__(self, log_dir, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log episode statistics when episode ends
        if self.locals.get('dones')[0]:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_reward = info['episode']['r']
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(info['episode']['l'])
                if 'success' in info:
                    self.episode_successes.append(info['success'])
                
                # Print each episode reward
                success_str = ""
                if 'success' in info:
                    success_str = f", Success: {info['success']}"
                print(f"Episode {len(self.episode_rewards)}: Reward: {episode_reward:.2f}{success_str}")
        
        return True
    
    def _on_training_end(self) -> None:
        # Save final metrics
        metrics_path = os.path.join(self.log_dir, 'training_metrics.npz')
        np.savez(metrics_path,
                 rewards=np.array(self.episode_rewards),
                 successes=np.array(self.episode_successes) if self.episode_successes else np.array([]),
                 lengths=np.array(self.episode_lengths))
        print(f"\nMetrics saved to {metrics_path}")
        
        # Print final statistics
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(f"Mean Reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
        if self.episode_successes:
            print(f"Success Rate: {np.mean(self.episode_successes):.2%}")
        print(f"Mean Episode Length: {np.mean(self.episode_lengths):.2f}")
        print("="*50)


def make_env(config, env_cls, task):
    """Create a single MetaWorld environment."""
    def _init():
        env = env_cls()
        env.set_task(task)
        return TimeLimit(env, max_episode_steps=config['horizon'])
    return _init


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a single PPO agent on MetaWorld')
    parser.add_argument('--task', '-t', type=str, default='pick-place-v3',
                        help='MetaWorld task name (default: pick-place-v3)')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--timesteps', type=int, default=1000000,
                        help='Total training timesteps (default: 1000000)')
    parser.add_argument('--log-dir', '-l', type=str, default='./single_agent_runs',
                        help='Directory for logs and checkpoints (default: ./single_agent_runs)')
    parser.add_argument('--task-index', type=int, default=0,
                        help='Task instance index from train_tasks (default: 0)')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    
    # Load configs
    env_config = get_config('./config/env/metaworld.yaml')
    alg_config = get_config('./config/algorithm/ppo_metaworld.yaml')
    
    # Override with command line args
    env_config['task'] = args.task
    alg_config['alg_seed'] = args.seed
    alg_config['total_source_timesteps'] = args.timesteps
    
    config = {**env_config, **alg_config}
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.task}_seed{args.seed}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    print("="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Task: {args.task}")
    print(f"Seed: {args.seed}")
    print(f"Total Timesteps: {args.timesteps:,}")
    print(f"Task Instance Index: {args.task_index}")
    print(f"Log Directory: {log_dir}")
    print("="*50)
    print()
    
    # Initialize MetaWorld environment
    print("Initializing MetaWorld environment...")
    ml1 = metaworld.ML1(env_name=args.task, seed=config['mw_seed'])
    
    # Get environment class and task instance
    env_name, env_cls = list(ml1.train_classes.items())[0]
    task_instance = ml1.train_tasks[args.task_index]
    
    print(f"Environment: {env_name}")
    print(f"Task Instance: {args.task_index}")
    print()
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(config, env_cls, task_instance)])
    
    # Initialize PPO agent
    print("Initializing PPO agent...")
    model = PPO(
        policy=config['policy'],
        env=env,
        learning_rate=3e-4,  # Standard learning rate
        n_steps=2048,  # More steps per update
        batch_size=64,  # Smaller batch for better gradients
        n_epochs=10,  # Standard number of epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Increased exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        verbose=0,
        tensorboard_log=log_dir
    )
    
    print("Model initialized successfully!")
    print()
    
    # Train the agent
    start_time = datetime.now()
    print(f"Training started at {start_time}")
    print()
    
    callback = MetricsCallback(log_dir)
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="PPO",
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    print()
    print(f"Training ended at {end_time}")
    print(f"Elapsed time: {elapsed}")
    print()
    
    # Save the trained model
    model_path = os.path.join(log_dir, 'final_model.zip')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate the trained agent
    print()
    print("Evaluating trained agent...")
    eval_episodes = 100
    eval_rewards = []
    eval_successes = []
    
    obs = env.reset()
    for episode in range(eval_episodes):
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            if done[0]:
                eval_rewards.append(episode_reward)
                if 'success' in info[0]:
                    eval_successes.append(info[0]['success'])
                obs = env.reset()
                break
    
    print()
    print("="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Episodes: {eval_episodes}")
    print(f"Mean Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    if eval_successes:
        print(f"Success Rate: {np.mean(eval_successes):.2%}")
    print("="*50)
    
    # Save evaluation results
    eval_path = os.path.join(log_dir, 'evaluation_results.npz')
    np.savez(eval_path,
             rewards=np.array(eval_rewards),
             successes=np.array(eval_successes) if eval_successes else np.array([]))
    print(f"Evaluation results saved to {eval_path}")
    
    env.close()
    print("\nDone!")


if __name__ == '__main__':
    main()

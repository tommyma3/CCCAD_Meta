"""
Compare performance between single PPO agent and Algorithm Distillation model.
This script loads metrics from both approaches and generates comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def load_single_agent_metrics(path):
    """Load metrics from single agent training."""
    data = np.load(path)
    return {
        'rewards': data['rewards'],
        'successes': data.get('successes', np.array([])),
        'lengths': data.get('lengths', np.array([]))
    }


def load_ad_metrics(path):
    """Load metrics from AD evaluation."""
    data = np.load(path)
    return {
        'rewards': data['rewards'] if 'rewards' in data else data,
        'successes': data.get('successes', np.array([]))
    }


def compute_moving_average(data, window=100):
    """Compute moving average."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_comparison(single_agent_data, ad_data, output_path=None):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Training curve (single agent) vs final AD performance
    ax = axes[0, 0]
    sa_rewards = single_agent_data['rewards']
    sa_smoothed = compute_moving_average(sa_rewards, window=100)
    
    ax.plot(sa_smoothed, label='Single Agent (PPO)', alpha=0.8)
    if len(ad_data['rewards']) > 0:
        ad_mean = np.mean(ad_data['rewards'])
        ax.axhline(ad_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'AD Mean: {ad_mean:.2f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (Moving Avg)')
    ax.set_title('Learning Curve Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Reward distribution comparison
    ax = axes[0, 1]
    if len(ad_data['rewards']) > 0:
        ax.hist(sa_rewards[-1000:], bins=30, alpha=0.5, label='Single Agent (last 1000)', 
                density=True, color='blue')
        ax.hist(ad_data['rewards'].flatten(), bins=30, alpha=0.5, label='AD', 
                density=True, color='red')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Density')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Success rate over time (single agent)
    ax = axes[1, 0]
    if len(single_agent_data['successes']) > 0:
        sa_success = single_agent_data['successes']
        sa_success_smoothed = compute_moving_average(sa_success.astype(float), window=100)
        ax.plot(sa_success_smoothed, label='Single Agent', color='blue')
        
        if len(ad_data['successes']) > 0:
            ad_success_rate = np.mean(ad_data['successes'])
            ax.axhline(ad_success_rate, color='red', linestyle='--', linewidth=2,
                      label=f'AD: {ad_success_rate:.2%}')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate (Moving Avg)')
        ax.set_title('Success Rate Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Compute statistics
    sa_final_rewards = sa_rewards[-100:] if len(sa_rewards) >= 100 else sa_rewards
    sa_mean = np.mean(sa_final_rewards)
    sa_std = np.std(sa_final_rewards)
    
    if len(ad_data['rewards']) > 0:
        ad_rewards_flat = ad_data['rewards'].flatten()
        ad_mean = np.mean(ad_rewards_flat)
        ad_std = np.std(ad_rewards_flat)
    else:
        ad_mean = ad_std = 0
    
    stats_text = "PERFORMANCE SUMMARY\n" + "="*40 + "\n\n"
    stats_text += "Single Agent (PPO, last 100 episodes):\n"
    stats_text += f"  Mean Reward: {sa_mean:.2f} ± {sa_std:.2f}\n"
    if len(single_agent_data['successes']) > 0:
        sa_success = np.mean(single_agent_data['successes'][-100:])
        stats_text += f"  Success Rate: {sa_success:.2%}\n"
    stats_text += f"  Total Episodes: {len(sa_rewards)}\n"
    stats_text += "\n"
    
    if len(ad_data['rewards']) > 0:
        stats_text += "Algorithm Distillation:\n"
        stats_text += f"  Mean Reward: {ad_mean:.2f} ± {ad_std:.2f}\n"
        if len(ad_data['successes']) > 0:
            ad_success = np.mean(ad_data['successes'])
            stats_text += f"  Success Rate: {ad_success:.2%}\n"
        stats_text += f"  Total Episodes: {len(ad_rewards_flat)}\n"
        stats_text += "\n"
        
        # Comparison
        stats_text += "="*40 + "\n"
        reward_diff = ad_mean - sa_mean
        stats_text += f"Reward Difference: {reward_diff:+.2f}\n"
        if len(single_agent_data['successes']) > 0 and len(ad_data['successes']) > 0:
            success_diff = ad_success - sa_success
            stats_text += f"Success Diff: {success_diff:+.2%}\n"
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare Single Agent vs Algorithm Distillation')
    parser.add_argument('--single-agent', '-s', type=str, required=True,
                        help='Path to single agent training_metrics.npz')
    parser.add_argument('--ad-model', '-a', type=str, required=True,
                        help='Path to AD evaluation results (.npy or .npz)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for comparison plot (optional)')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    
    # Check if files exist
    if not os.path.exists(args.single_agent):
        print(f"Error: Single agent file not found: {args.single_agent}")
        return
    
    if not os.path.exists(args.ad_model):
        print(f"Error: AD model file not found: {args.ad_model}")
        return
    
    print("Loading data...")
    print(f"  Single Agent: {args.single_agent}")
    print(f"  AD Model: {args.ad_model}")
    print()
    
    # Load data
    single_agent_data = load_single_agent_metrics(args.single_agent)
    ad_data = load_ad_metrics(args.ad_model)
    
    print("Data loaded successfully!")
    print(f"  Single Agent Episodes: {len(single_agent_data['rewards'])}")
    print(f"  AD Evaluations: {len(ad_data['rewards'].flatten())}")
    print()
    
    # Generate comparison plot
    print("Generating comparison plot...")
    plot_comparison(single_agent_data, ad_data, args.output)
    
    print("Done!")


if __name__ == '__main__':
    main()

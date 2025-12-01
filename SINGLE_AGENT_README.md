# Single Agent Training - MetaWorld

This guide explains how to train and evaluate a single PPO agent on a single MetaWorld environment to understand the source algorithm's behavior before applying Algorithm Distillation.

## Overview

These scripts allow you to:
1. Train a PPO agent on a single MetaWorld task instance
2. Evaluate the trained agent's performance
3. Visualize how the source algorithm (PPO) learns and performs

This is useful for:
- Understanding baseline performance before applying Algorithm Distillation
- Debugging environment setup
- Quick experimentation with different tasks
- Validating that PPO can solve the task

## Files

- `train_single_agent.py` - Train a single PPO agent
- `evaluate_single_agent.py` - Evaluate a trained agent

## Quick Start

### Training

Train a PPO agent on pick-place-v2:

```bash
python train_single_agent.py
```

With custom settings:

```bash
python train_single_agent.py --task reach-v2 --timesteps 500000 --seed 42
```

### Evaluation

Evaluate a trained agent:

```bash
python evaluate_single_agent.py --model-path ./single_agent_runs/pick-place-v2_seed0_20251201_120000/final_model.zip
```

With custom settings:

```bash
python evaluate_single_agent.py \
    --model-path ./single_agent_runs/reach-v2_seed42_20251201_120000/final_model.zip \
    --task reach-v2 \
    --episodes 200 \
    --task-index 0
```

## Training Script Options

```
python train_single_agent.py [OPTIONS]

Options:
  --task, -t TEXT         MetaWorld task name (default: pick-place-v2)
  --seed, -s INT          Random seed (default: 0)
  --timesteps INT         Total training timesteps (default: 1000000)
  --log-dir, -l TEXT      Directory for logs (default: ./single_agent_runs)
  --task-index INT        Task instance index from train_tasks (default: 0)
```

### Example Training Commands

**Quick test (100k steps):**
```bash
python train_single_agent.py --timesteps 100000
```

**Train on reach-v2 (easier task):**
```bash
python train_single_agent.py --task reach-v2 --timesteps 500000
```

**Train on push-v2:**
```bash
python train_single_agent.py --task push-v2 --timesteps 1000000
```

**Train on pick-place-v2 (harder task):**
```bash
python train_single_agent.py --task pick-place-v2 --timesteps 2000000
```

**Different task instance:**
```bash
python train_single_agent.py --task-index 5 --seed 123
```

## Evaluation Script Options

```
python evaluate_single_agent.py [OPTIONS]

Required:
  --model-path, -m TEXT   Path to trained model (.zip file)

Optional:
  --task, -t TEXT         MetaWorld task name (default: pick-place-v2)
  --episodes, -e INT      Number of evaluation episodes (default: 100)
  --task-index INT        Task instance index (default: 0)
  --render, -r            Render the environment during evaluation
  --deterministic, -d     Use deterministic actions (default: True)
```

### Example Evaluation Commands

**Basic evaluation:**
```bash
python evaluate_single_agent.py -m ./single_agent_runs/pick-place-v2_seed0_20251201_120000/final_model.zip
```

**More episodes:**
```bash
python evaluate_single_agent.py \
    -m ./single_agent_runs/pick-place-v2_seed0_20251201_120000/final_model.zip \
    --episodes 500
```

**Evaluate on different task instance:**
```bash
python evaluate_single_agent.py \
    -m ./single_agent_runs/pick-place-v2_seed0_20251201_120000/final_model.zip \
    --task-index 10
```

**With rendering (slower):**
```bash
python evaluate_single_agent.py \
    -m ./single_agent_runs/reach-v2_seed0_20251201_120000/final_model.zip \
    --render
```

## Output Files

After training, you'll find in the log directory:

```
single_agent_runs/
  pick-place-v2_seed0_20251201_120000/
    config.yaml                    # Training configuration
    final_model.zip                # Trained PPO model
    training_metrics.npz           # Training episode rewards, successes, lengths
    evaluation_results.npz         # Initial evaluation results
    PPO_1/                         # TensorBoard logs
      events.out.tfevents...
```

After evaluation:

```
single_agent_runs/
  pick-place-v2_seed0_20251201_120000/
    ...
    eval_task0_episodes100.npz     # Evaluation results for task 0
```

## Monitoring Training

### Terminal Output

During training, you'll see:
- Episode progress every 10 episodes
- Mean reward and success rate over last 100 episodes
- TensorBoard logging information

### TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir=./single_agent_runs
```

Then open http://localhost:6006 in your browser.

Metrics available:
- `rollout/ep_rew_mean` - Mean episode reward
- `rollout/ep_len_mean` - Mean episode length
- `train/policy_loss` - Policy loss
- `train/value_loss` - Value function loss
- `train/entropy_loss` - Entropy bonus

## Understanding Results

### Training Metrics

The training will save `training_metrics.npz` containing:
- `rewards` - Reward for each episode
- `successes` - Success indicator for each episode
- `lengths` - Length of each episode

Load and analyze:

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load('single_agent_runs/pick-place-v2_seed0_20251201_120000/training_metrics.npz')
rewards = data['rewards']
successes = data['successes']

# Plot learning curve
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards')

plt.subplot(1, 2, 2)
window = 100
success_rate = np.convolve(successes, np.ones(window)/window, mode='valid')
plt.plot(success_rate)
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.title(f'Success Rate (window={window})')
plt.tight_layout()
plt.show()
```

### Evaluation Results

Evaluation saves `evaluation_results.npz` containing:
- `rewards` - Reward for each evaluation episode
- `successes` - Success indicator for each evaluation episode

## Task Difficulty Guidelines

Approximate training times and expected performance:

| Task | Timesteps | Success Rate | Notes |
|------|-----------|--------------|-------|
| reach-v2 | 500k | ~90%+ | Easiest, good for testing |
| push-v2 | 1M | ~70-80% | Medium difficulty |
| pick-place-v2 | 2M | ~50-70% | Harder, requires precise control |
| door-open-v2 | 1-2M | ~60-80% | Medium-hard |
| drawer-close-v2 | 1-2M | ~60-80% | Medium-hard |

## Tips

1. **Start with reach-v2**: It's the easiest task and trains quickly. Use it to verify everything works.

2. **Watch for convergence**: If success rate isn't improving after 500k steps, something may be wrong.

3. **Task instances matter**: Each task has 50 different instances. Some are easier than others. Try different `--task-index` values.

4. **Hyperparameters**: The default PPO hyperparameters from `config/algorithm/ppo_metaworld.yaml` are used. They're tuned for data collection but may not be optimal for single-agent training.

5. **GPU usage**: PPO will automatically use GPU if available. Training is much faster on GPU.

6. **Training time**: 
   - 1M steps ≈ 1-3 hours on GPU
   - 1M steps ≈ 5-10 hours on CPU

7. **Comparison with Algorithm Distillation**: After training, compare this agent's performance with the AD model to see the benefit of in-context learning.

## Troubleshooting

**Low success rate:**
- Try training longer
- Try different task instances (some are easier)
- Try easier tasks first (reach-v2)
- Check TensorBoard for training stability

**Out of memory:**
- Reduce `n_steps` in config (default: 100)
- Reduce `batch_size` in config (default: 200)

**Slow training:**
- Ensure GPU is being used (check with `nvidia-smi`)
- Reduce `n_epochs` in config (default: 20)

**Import errors:**
- Make sure all dependencies are installed: `uv add metaworld stable-baselines3`

## Next Steps

After training a single agent:

1. **Compare with Algorithm Distillation**: See how AD performs vs. this single-task agent
2. **Collect trajectories**: Use this agent to collect data for AD training
3. **Try different tasks**: Understand which tasks are easier/harder
4. **Experiment with hyperparameters**: Tune PPO for better performance

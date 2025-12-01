# Quick Start Guide: Using MetaWorld

## Prerequisites
```bash
# Make sure metaworld is installed
uv add metaworld
```

## Two Approaches

This project supports two ways to work with MetaWorld:

1. **Single Agent Training** - Train a single PPO agent on one task instance (see `SINGLE_AGENT_README.md`)
2. **Algorithm Distillation** - Train AD model from multiple PPO trajectories (this guide)

### When to use Single Agent Training?

Use `train_single_agent.py` to:
- Understand baseline PPO performance on MetaWorld
- Debug environment setup quickly
- Test if a task is learnable
- Get quick results without data collection

See `SINGLE_AGENT_README.md` for detailed instructions.

## Configuration Files

The project now has separate config files for darkroom and metaworld:

### MetaWorld:
- Environment: `config/env/metaworld.yaml`
- Algorithm: `config/algorithm/ppo_metaworld.yaml`
- Model (Standard AD): `config/model/ad_metaworld.yaml`
- Model (Compressed AD): `config/model/ad_compressed_metaworld.yaml`

### Darkroom (Original):
- Environment: `config/env/darkroom.yaml`
- Algorithm: `config/algorithm/ppo_darkroom.yaml`
- Model: `config/model/ad_dr.yaml`

## Workflow

### 1. Data Collection

Collect training and test trajectories using PPO:

```bash
python collect.py
```

This will create:
```
datasets/
  pick-place-v2/
    history_pick-place-v2_PPO_alg-seed0.hdf5  # 50 train tasks
    test/
      history_pick-place-v2_PPO_alg-seed0.hdf5  # 10 test tasks
```

**Note**: Data collection takes significant time (~hours for 50 tasks).

### 2. Training

Train the Algorithm Distillation model:

```bash
python train.py
```

By default, this will:
- Load config from `config/env/metaworld.yaml`, `config/algorithm/ppo_metaworld.yaml`, and `config/model/ad_dr.yaml`
- Train on pick-place-v2 task
- Save checkpoints to `./runs/AD-metaworld-pick-place-v2/` or `./runs/CompressedAD-metaworld-pick-place-v2/`

**To use CompressedAD model**, modify train.py line 32:
```python
config.update(get_config('./config/model/ad_compressed_metaworld.yaml'))
```

### 3. Evaluation

Evaluate the trained model:

```bash
python evaluate.py
```

**Important**: Update the checkpoint directory in evaluate.py (line 23):
```python
ckpt_dir = './runs/CompressedAD-metaworld-pick-place-v2'  # Update this
```

Results will be saved as:
- `eval_result_reward.npy`: Episode rewards
- `eval_result_success.npy`: Success rates

## Changing Tasks

To train on a different MetaWorld task, edit `config/env/metaworld.yaml`:

```yaml
task: reach-v2  # Change to desired task
```

Available ML1 tasks include:
- reach-v2
- push-v2
- pick-place-v2
- peg-insert-side-v2
- door-open-v2
- drawer-close-v2
- button-press-topdown-v2
- ... and many more

## Model Selection

### Standard AD Model
```python
# In train.py, line 32:
config.update(get_config('./config/model/ad_metaworld.yaml'))
```

### Compressed AD Model (with hierarchical compression)
```python
# In train.py, line 32:
config.update(get_config('./config/model/ad_compressed_metaworld.yaml'))
```

The Compressed AD model includes:
- Automatic compression when context exceeds max length
- Hierarchical compression (up to 3 levels)
- Encoder that compresses sequences into latent representations
- Training with variable compression depths (0-3 cycles)

## Key Parameters

### Environment (config/env/metaworld.yaml)
- `task`: ML1 task name
- `horizon`: Episode length (100 for most tasks)
- `dim_obs`: Observation dimension (11 for metaworld)
- `dim_actions`: Action dimension (4 for metaworld)

### Training (config/model/*.yaml)
- `train_batch_size`: Batch size for training (512)
- `train_timesteps`: Total training steps (50000)
- `train_source_timesteps`: Source trajectory length (10000)
- `n_transit`: Context length for transformer (1000)

### Compression (config/model/ad_compressed_metaworld.yaml)
- `n_latent`: Number of latent tokens (60)
- `max_compression_depth`: Maximum compression levels (3)
- `min_compress_length`: Minimum segment length to compress (10)
- `max_compress_length`: Maximum segment length to compress (50)

## Tips

1. **Data Collection**: Start with a smaller subset (e.g., 10 tasks) for testing before running full 50 tasks.

2. **Training Time**: Training 50k steps typically takes several hours on GPU.

3. **Evaluation**: The evaluation uses 50 test environments and runs for many episodes (100-2000 depending on task), so it can take significant time.

4. **Debugging**: Use smaller values in config files for quick testing:
   - `train_timesteps: 1000`
   - `train_source_timesteps: 1000`
   - Fewer tasks in collect.py

5. **Monitor Training**: TensorBoard logs are saved in the run directory:
   ```bash
   tensorboard --logdir=./runs
   ```

# Migration from Darkroom to MetaWorld - Summary

This document summarizes the changes made to migrate your Algorithm Distillation with compression implementation from the Darkroom environment to MetaWorld environments.

## Overview

Your project has been successfully migrated to use MetaWorld ML1 environments. The key changes involve:
1. Environment setup and data collection
2. Dataset loading with proper observation dimensions
3. Training and evaluation loops
4. Configuration files

## Files Modified

### 1. Configuration Files (NEW)
- **config/env/metaworld.yaml**: MetaWorld environment configuration
  - Task: pick-place-v2 (configurable)
  - Observation dimension: 11
  - Action dimension: 4
  - Horizon: 100 steps

- **config/algorithm/ppo_metaworld.yaml**: PPO algorithm configuration for MetaWorld
  - Training timesteps: 1M
  - Batch size: 200
  - Number of processes: 8

- **config/model/ad_metaworld.yaml**: Standard AD model config for MetaWorld
- **config/model/ad_compressed_metaworld.yaml**: Compressed AD model config for MetaWorld
  - Includes compression parameters (n_latent, max_compression_depth, etc.)

### 2. env/__init__.py
**Changes:**
- Added `metaworld` import and `TimeLimit` wrapper
- Modified `make_env()` function to support both darkroom and metaworld
- For metaworld: requires `env_cls` and `task` parameters
- For darkroom: uses config['env'] and kwargs (backward compatible)

### 3. collect.py
**Major Rewrite:**
- Switched from darkroom goal-based collection to MetaWorld ML1 task-based collection
- Uses `metaworld.ML1()` to get train and test tasks
- Collects data for both train tasks (50 by default) and test tasks (10 by default)
- Data saved in structured directories: `datasets/{task}/` and `datasets/{task}/test/`
- Worker function now accepts `env_cls` and `task_instance` instead of `goal`

### 4. dataset.py
**Changes to ADDataset:**
- Added `n_seed` parameter (replaces environment indexing for metaworld)
- Supports both darkroom (original) and metaworld data loading
- For metaworld:
  - Loads from task-based directory structure
  - Extracts only observation dimensions (first `dim_obs` elements)
  - Actions are 4-dimensional continuous for metaworld

**Changes to ADCompressedDataset:**
- Similar changes as ADDataset
- Maintains all compression logic for hierarchical compression
- Supports variable compression depths (0-3 cycles)

### 5. utils.py
**Changes:**
- Removed darkroom-specific imports (`map_dark_states`, `F.one_hot`)
- Simplified `ad_collate_fn()` to work generically with continuous actions
- Removed `partial()` and `grid_size` dependency
- Now works for both darkroom and metaworld seamlessly

### 6. train.py
**Changes:**
- Config loading changed to use `metaworld.yaml` and `ppo_metaworld.yaml`
- Log directory includes task name instead of env_split_seed
- Dataset initialization uses `n_seed` parameter:
  - Train: 50 seeds
  - Test: 10 seeds
- Environment setup:
  - Uses `metaworld.ML1()` to get tasks
  - Creates 20 test environments (10 train + 10 test tasks)
  - Uses proper `make_env()` with `env_cls` and `task`

### 7. evaluate.py
**Complete Rewrite:**
- Checkpoint directory points to metaworld runs
- Uses `metaworld.ML1()` for test environment creation
- Creates 50 test environments from test tasks
- Evaluation episodes based on task difficulty:
  - reach-v2: 100 episodes
  - push-v2: 300 episodes
  - pick-place-v2, peg-insert-side-v2: 2000 episodes
  - Others: 200 episodes
- Saves both reward and success metrics

## Key Differences: Darkroom vs MetaWorld

| Aspect | Darkroom | MetaWorld |
|--------|----------|-----------|
| Environment Type | Grid-based navigation | Robotics manipulation |
| Observation | 2D position (x, y) | 11D continuous state |
| Action Space | 5 discrete actions | 4D continuous actions |
| Task Definition | Goal position | Task instances from ML1 |
| Data Organization | Flat HDF5 file | Task-based directories |
| Episode Length | 20 steps | 100 steps |
| Train/Test Split | Env split seed | ML1 train/test tasks |

## Usage

### Data Collection
```bash
python collect.py
```
This will:
1. Create datasets/{task}/ directory
2. Collect trajectories for train tasks
3. Collect trajectories for test tasks in datasets/{task}/test/

### Training
```bash
python train.py
```
The script will automatically:
1. Load metaworld config files
2. Load dataset from datasets/{task}/
3. Train AD or CompressedAD model
4. Save checkpoints to ./runs/{model}-metaworld-{task}/

### Evaluation
```bash
python evaluate.py
```
Update the `ckpt_dir` variable to point to your checkpoint directory.

## Model Configuration

You can switch between:
1. **Standard AD**: Use `config/model/ad_metaworld.yaml`
2. **Compressed AD**: Use `config/model/ad_compressed_metaworld.yaml`

To change the task, edit `config/env/metaworld.yaml` and set `task` to one of:
- reach-v2
- push-v2
- pick-place-v2
- peg-insert-side-v2
- etc. (any ML1 task)

## Notes

1. **Backward Compatibility**: The darkroom environment still works! All darkroom-specific code is preserved in conditional branches.

2. **Compression**: The CompressedAD model maintains all its hierarchical compression features when working with metaworld.

3. **Data Format**: MetaWorld data includes more dimensions in observations (39 total, but only first 11 are used for observations).

4. **Multiprocessing**: Data collection uses multiprocessing with 8 processes by default (configurable in config).

5. **Success Metric**: MetaWorld provides a success signal which is now tracked during evaluation.

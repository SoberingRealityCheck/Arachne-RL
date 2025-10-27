# Training Usage Guide

The training script now uses command-line arguments instead of interactive prompts.

## Basic Usage

```bash
# Train with default settings (simple_quadruped, headless, new model)
python train.py

# Train with GUI
python train.py --gui

# Continue training from specific model
python train.py --model models/quadruped_checkpoints/quadruped_model_1000000_steps.zip

# Train specific robot
python train.py --robot servobot

# Train with custom timesteps
python train.py --timesteps 5000000
```

## All Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--robot` | Robot to train (`simple_quadruped`, `servobot`, `arachne`) | `simple_quadruped` |
| `--gui` | Run with PyBullet GUI | `False` (headless) |
| `--model` | Model file to load for continued training | `None` (create new) |
| `--timesteps` | Total training timesteps | `2000000` |
| `--target-speed` | Target speed for robot | `1.0` |
| `--learning-rate` | PPO learning rate | `0.0001` |

## Examples

```bash
# Train servobot with GUI for 5M timesteps
python train.py --robot servobot --gui --timesteps 5000000

# Continue training from existing model
python train.py --model models/servobot_checkpoints/servobot_model_500000_steps.zip

# Train with custom learning rate and target speed
python train.py --learning-rate 0.0003 --target-speed 0.5

# Continue training arachne with GUI
python train.py --robot arachne --model models/arachne_checkpoints/current/arachne_model_1000000_steps.zip --gui
```

## Quick Reference

**New model, headless:**
```bash
python train.py
```

**Continue training from checkpoint:**
```bash
python train.py --model path/to/model.zip
```

**Different robot:**
```bash
python train.py --robot servobot
```

## Notes

- Specifying `--model` automatically loads that model for continued training
- Model paths can be absolute or relative to the robot's save directory
- Training progress is saved automatically every 100k steps to `models/{robot}/current/`


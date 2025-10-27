# This script sets up a PyBullet-based reinforcement learning environment for a
# simple quadruped robot. The task is to reach and touch a green target box
# (specified by its center and size) within one episode (~30s by default),
# while staying upright and avoiding jumping.
#
# Dependencies:
#   pip install pybullet gymnasium stable-baselines3[extra]
#
# Notes:
# - The URDF file 'simple_quadruped.urdf' must be in the same directory.
# - Uses Stable-Baselines3 PPO as the baseline RL agent.

import os
import time
import math
import argparse
import numpy as np
import pybullet as py
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import pandas as pd
import matplotlib.pyplot as plt

# These are our custom modules: 
# utils has a bunch of helper functions for importing stuff and navigating local files
# env has our custom BaseEnv envrironment class that inherits from gym.Env and implements the RL environment
from src.envs import env
from src.utils import utils
from src.utils.plotting_callback import LivePlottingCallback, LivePlottingCallbackNoGUI
from src.utils.config import ROBOTS

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a quadruped robot using PPO')
    parser.add_argument('--robot', type=str, default='simple_quadruped',
                        choices=list(ROBOTS.keys()),
                        help='Robot to train (default: simple_quadruped)')
    parser.add_argument('--gui', action='store_true',
                        help='Run with PyBullet GUI (default: headless)')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model file to load for continued training (optional, default: create new model)')
    parser.add_argument('--timesteps', type=int, default=2000000,
                        help='Total training timesteps (default: 2000000)')
    parser.add_argument('--target-speed', type=float, default=1.0,
                        help='Target speed for the robot (default: 1.0)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate for PPO (default: 0.0001)')
    
    args = parser.parse_args()

    # Set render mode based on GUI flag
    render_mode = 'human' if args.gui else 'headless'
    
    # Get robot configuration
    robot_name = args.robot
    if robot_name not in ROBOTS:
        raise ValueError(f"Robot '{robot_name}' not found in configuration!")
    
    urdf_file = ROBOTS[robot_name]['urdf_file']
    save_path = ROBOTS[robot_name]['save_path']
    save_prefix = ROBOTS[robot_name]['save_prefix']
    
    print(f"\n{'='*50}")
    print(f"Training Configuration:")
    print(f"{'='*50}")
    print(f"Robot: {robot_name}")
    print(f"URDF: {urdf_file}")
    print(f"Save Path: {save_path}")
    print(f"Render Mode: {render_mode}")
    print(f"Target Speed: {args.target_speed}")
    print(f"Total Timesteps: {args.timesteps}")
    print(f"{'='*50}\n")

    # Pass box parameters into the environment.
    min_z = env.get_min_z(urdf_file)
    env = env.BaseEnv(
        render_mode=render_mode, 
        urdf_filename=urdf_file, 
        start_position=[0, 0, -min_z],
        target_speed=args.target_speed,
    )
    
    # Handle model loading - if --model is specified, load it; otherwise create new
    if args.model:
        # Specific model file provided
        model_path = args.model
        if not os.path.exists(model_path):
            # Try relative to save_path
            model_path = os.path.join(save_path, model_path)
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            exit(1)
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env, device='cpu')
    else:
        # Create new model with stable training parameters
        print("Creating new model...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            n_steps=2048,
            learning_rate=args.learning_rate,
            batch_size=64,         # Smaller batches for better gradient estimates
            target_kl=0.015,       # Early stop if KL gets too high (prevents spikes)
            ent_coef=0.01,        # Slight exploration bonus
        )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=os.path.join(save_path, "current/"),
        name_prefix=save_prefix
    )
    
    # Add live plotting callback (different behavior for GUI vs headless)
    if render_mode == 'human':
        # With GUI: Show live updating plots
        plot_callback = LivePlottingCallback(
            plot_freq=2048,  # Update every iteration (n_steps)
            max_points=500,  # Keep last 500 data points for performance
            verbose=1
        )
        print("\n✓ Live plotting enabled! A plot window will open showing real-time metrics.")
        print("  The plot updates every 2048 steps (~7 seconds at 290 fps)")
    else:
        # Headless: Save plots periodically to files
        plot_callback = LivePlottingCallbackNoGUI(
            plot_freq=2048,      # Collect data every iteration
            save_freq=50000,     # Save plot image every 50k steps
            save_path='./training_plots/',
            verbose=1
        )
        print("\n✓ Plot saving enabled! Training plots will be saved to ./training_plots/")
        print("  Plots saved every 50k steps")
    
    # Combine callbacks
    callback_list = CallbackList([checkpoint_callback, plot_callback])

    print(f"\nStarting training for {args.timesteps} timesteps...")
    print("Press Ctrl+C to stop training early.\n")

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback_list, progress_bar=True)
    except KeyboardInterrupt:
        print("Training stopped by user.")
        reward_history = pd.read_csv(env.reward_history_filename)
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 12))
        for col, ax in zip(reward_history.columns, axes.flatten()):
            ax.plot(reward_history[col])
            ax.set_title(f"Reward History - {col}")
            ax.set_xlabel(col) # Set the x-axis label
            ax.set_ylabel('value')
        plt.tight_layout()
        plt.show()
    finally:
        env.close()
    
    print("Training finished.")

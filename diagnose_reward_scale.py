"""
Emergency Diagnostic Script - Check if rewards are correctly scaled
Run this to verify your reward function is working properly
"""

import numpy as np
import sys
import os

from src.envs.env import BaseEnv
from src.utils.utils import select_robot

print("="*70)
print("EMERGENCY REWARD SCALE DIAGNOSTIC")
print("="*70)
print()

# Create environment
robot_choice = 'robots/arachne/arachne.urdf'
env = BaseEnv(urdf_filename=robot_choice, render_mode=None)

print(f"Testing reward function for {robot_choice}...")
print()

# Reset environment
obs, info = env.reset()

# Take 10 random actions and record rewards
rewards = []
episode_reward = 0
step_count = 0

print("Taking 10 random actions and measuring rewards:")
print("-" * 70)

for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    rewards.append(reward)
    episode_reward += reward
    step_count += 1
    
    print(f"Step {i+1}: reward = {reward:.2f}")
    
    if terminated or truncated:
        print(f"  (Episode ended)")
        break

print("-" * 70)
print()

# Calculate statistics
avg_reward = np.mean(rewards)
max_reward = np.max(rewards)
min_reward = np.min(rewards)

print("RESULTS:")
print(f"  Average reward per step: {avg_reward:.2f}")
print(f"  Max reward seen: {max_reward:.2f}")
print(f"  Min reward seen: {min_reward:.2f}")
print(f"  Episode reward so far: {episode_reward:.2f}")
print()

# Expected values
expected_max = 21.52
expected_typical = 10.0

print("EXPECTED VALUES (with correct reward scale):")
print(f"  Theoretical max per step: ~{expected_max:.2f}")
print(f"  Typical per step (early training): ~{expected_typical:.2f}")
print()

print("="*70)
print("DIAGNOSIS:")
print("="*70)

if max_reward > 50:
    print("🚨 CRITICAL: Rewards are WAY TOO HIGH!")
    print(f"   Max reward {max_reward:.2f} >> expected {expected_max:.2f}")
    print()
    print("   This means rewards are still being accumulated incorrectly!")
    print("   Possible causes:")
    print("     1. Reward calculated in physics loop (10x accumulation)")
    print("     2. Survival reward still scaled by steps_taken")
    print("     3. Some other accumulation bug")
    print()
    print("   ACTION: Fix the reward calculation in src/envs/env.py")
    
elif max_reward > 30:
    print("⚠️  WARNING: Rewards seem higher than expected")
    print(f"   Max reward {max_reward:.2f} > expected {expected_max:.2f}")
    print()
    print("   This might cause training issues.")
    print("   Check reward components are balanced correctly.")
    
elif max_reward < 5:
    print("⚠️  WARNING: Rewards seem very low")
    print(f"   Max reward {max_reward:.2f} << expected {expected_max:.2f}")
    print()
    print("   Robot might not be getting enough signal to learn.")
    print("   Check that velocity/orientation rewards are working.")
    
else:
    print("✅ Reward scale looks reasonable!")
    print(f"   Max reward {max_reward:.2f} is in expected range")
    print()
    print("   If training is still failing, the issue is likely:")
    print("     - Learning rate too high")
    print("     - Value function architecture")
    print("     - Task is too difficult")

print("="*70)
print()

# Estimate what episode reward should be
estimated_ep_reward = avg_reward * 120  # Typical episode length
print(f"ESTIMATED EPISODE REWARD (120 steps): ~{estimated_ep_reward:.0f}")
print()

if estimated_ep_reward > 10000:
    print("🚨 Episode rewards will be ~{:.0f}".format(estimated_ep_reward))
    print("   This is TOO HIGH for value function to learn!")
    print("   Explained variance will stay at 0")
    print("   KL divergence will spike uncontrollably")
    print()
    print("   YOU MUST FIX THE REWARD SCALE BEFORE TRAINING")
elif estimated_ep_reward > 5000:
    print("⚠️  Episode rewards will be ~{:.0f}".format(estimated_ep_reward))
    print("   This might be too high.")
    print("   Monitor explained variance closely.")
else:
    print("✅ Episode rewards should be ~{:.0f}".format(estimated_ep_reward))
    print("   This is a reasonable scale for PPO to learn.")

env.close()

print()
print("="*70)
print("Diagnostic complete!")
print("="*70)

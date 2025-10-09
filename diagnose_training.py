"""
Diagnostic script to analyze your trained model's behavior
Run this to see detailed reward breakdowns and identify issues
"""

from stable_baselines3 import PPO
import numpy as np
import pybullet as p
from src.envs.env import BaseEnv, get_min_z
from src.utils import utils
import matplotlib.pyplot as plt

def analyze_episode(env, model, num_steps=500):
    """Run one episode and collect detailed statistics"""
    obs, info = env.reset()
    
    stats = {
        'velocity_errors': [],
        'velocity_magnitudes': [],
        'target_velocity_magnitudes': [],
        'orientation_errors': [],
        'rewards': [],
        'forward_rewards': [],
        'orientation_rewards': [],
        'is_fallen': []
    }
    
    for step in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get current state
        base_vel, _ = p.getBaseVelocity(env.robot_id)
        current_base_pos, current_base_orient = p.getBasePositionAndOrientation(env.robot_id)
        
        # Calculate errors
        vel_error = np.linalg.norm(np.array(base_vel) - np.array(env.target_velocity))
        vel_mag = np.linalg.norm(base_vel)
        target_vel_mag = np.linalg.norm(env.target_velocity)
        
        # Calculate orientation error
        rot_matrix = p.getMatrixFromQuaternion(current_base_orient)
        forward_vector = np.array([-rot_matrix[3], rot_matrix[0], rot_matrix[6]])
        target_direction = env.target_velocity / (np.linalg.norm(env.target_velocity) + 1e-6)
        orientation_alignment = np.dot(forward_vector, target_direction)
        
        # Check if fallen
        local_up_vector = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
        is_fallen = current_base_pos[2] < 0.1 or local_up_vector[2] < 0.5
        
        stats['velocity_errors'].append(vel_error)
        stats['velocity_magnitudes'].append(vel_mag)
        stats['target_velocity_magnitudes'].append(target_vel_mag)
        stats['orientation_errors'].append(1.0 - orientation_alignment)  # Error: 0 is perfect
        stats['rewards'].append(reward)
        stats['is_fallen'].append(is_fallen)
        
        if terminated or truncated:
            break
    
    return stats

def plot_diagnostics(stats):
    """Create diagnostic plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Velocity tracking
    axes[0, 0].plot(stats['velocity_magnitudes'], label='Actual Velocity Magnitude', alpha=0.7)
    axes[0, 0].plot(stats['target_velocity_magnitudes'], label='Target Velocity Magnitude', alpha=0.7)
    axes[0, 0].set_title('Velocity Magnitude Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Velocity (m/s)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Velocity error
    axes[0, 1].plot(stats['velocity_errors'], color='red', alpha=0.7)
    axes[0, 1].set_title('Velocity Error Over Time')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Error (m/s)')
    axes[0, 1].grid(True)
    
    # Orientation error
    axes[1, 0].plot(stats['orientation_errors'], color='green', alpha=0.7)
    axes[1, 0].set_title('Orientation Error Over Time')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Error (1 - alignment)')
    axes[1, 0].grid(True)
    
    # Rewards
    axes[1, 1].plot(stats['rewards'], alpha=0.7)
    axes[1, 1].set_title('Reward Over Time')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_diagnostics.png')
    print("Diagnostic plot saved as 'training_diagnostics.png'")
    
    # Print statistics
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"Average Velocity Error: {np.mean(stats['velocity_errors']):.3f} m/s")
    print(f"Velocity Error Std Dev: {np.std(stats['velocity_errors']):.3f} m/s")
    print(f"Average Actual Velocity: {np.mean(stats['velocity_magnitudes']):.3f} m/s")
    print(f"Average Target Velocity: {np.mean(stats['target_velocity_magnitudes']):.3f} m/s")
    print(f"Average Orientation Error: {np.mean(stats['orientation_errors']):.3f}")
    print(f"Average Reward: {np.mean(stats['rewards']):.3f}")
    print(f"Episode Length: {len(stats['rewards'])} steps")
    print(f"Fell: {'Yes' if any(stats['is_fallen']) else 'No'}")
    print("="*60)

if __name__ == "__main__":
    # Select and load model
    urdf_file, save_path, save_prefix, model_path = utils.select_robot(load_model=True)
    
    min_z = get_min_z(urdf_file)
    env = BaseEnv(
        render_mode='human',  # Use 'headless' for faster analysis
        urdf_filename=urdf_file, 
        start_position=[0, 0, -min_z],
        target_speed=0.5
    )
    
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env, device='cpu')
    
    print("Running diagnostic episode...")
    stats = analyze_episode(env, model, num_steps=1000)
    
    plot_diagnostics(stats)
    
    env.close()

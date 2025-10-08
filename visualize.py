
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import numpy as np
import pybullet as p

from src.envs.env import BaseEnv, get_min_z
from src.utils import utils
from gymnasium import wrappers

'''
This script is a mismash of the quadreped run_trained.py and the servobot train.py scripts to load and run a trained servobot model
to demonstrate its movement in the pybullet gui. right now it looks pretty silly bc i just ran the training on my laptop for 15 minutes
but we could prolly get some better results if we ran it longer. 
'''

class VisualizationEnv(BaseEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
    def __init__(self, urdf_filename, start_position=[0,0,0], target_speed=0.5, render_mode='human'):
        super().__init__(render_mode=render_mode, urdf_filename=urdf_filename, start_position=start_position, target_speed=target_speed)
        
        # Set up a interactive debug variable for pybullet to control: 
        #   - target velocity direction (0 to 1 times 2pi)
        #   - target velocity magnitude (0 to 1)
        #   - Target orientation (0 to 1 times 2pi)
        self.target_velocity_direction_id = p.addUserDebugParameter("Target Velocity Direction", 0, 1, 0)
        self.target_velocity_magnitude_id = p.addUserDebugParameter("Target Velocity Magnitude", 0, 1, 0)
        self.target_orientation_id = p.addUserDebugParameter("Target Orientation", 0, 1, 0)

        # Initialize debug object lines to be drawn on for visualization of orientation/velocity
        self.debug_lines = []

    def reset(self, seed=None, options=None):
        """Override reset to use slider values instead of random targets"""
        # Call parent reset first
        obs, info = super().reset(seed=seed, options=options)
        
        # Immediately override the random targets with slider values
        # Read the current slider positions
        try:
            direction = p.readUserDebugParameter(self.target_velocity_direction_id) * 2 * 3.14159
            magnitude = p.readUserDebugParameter(self.target_velocity_magnitude_id)
            target_orientation_angle = p.readUserDebugParameter(self.target_orientation_id) * 2 * 3.14159
            
            # Set target velocity from sliders
            self.target_velocity = [magnitude * self.target_speed * np.cos(direction), 
                                   magnitude * self.target_speed * np.sin(direction), 0]
            # Set target orientation from slider (as quaternion)
            self.target_orientation = [0, 0, np.sin(target_orientation_angle / 2), np.cos(target_orientation_angle / 2)]
        except Exception as e:
            print(f"Warning: Could not read debug sliders in reset: {e}")
            # Keep the random values from parent reset
        
        return obs, info

    def step(self, action):
        # Read the debug parameters and set the target velocity and orientation accordingly
        try:
            direction = p.readUserDebugParameter(self.target_velocity_direction_id) * 2 * 3.14159  # 0 to 2pi
        except Exception as e:
            print(f"Error reading user debug parameter: {e}")
            direction = 0  # Default to 0 if there's an error
        try:
            magnitude = p.readUserDebugParameter(self.target_velocity_magnitude_id)  # 0 to 1
        except Exception as e:
            print(f"Error reading user debug parameter: {e}")
            magnitude = 0  # Default to 0 if there's an error
        try:
            target_orientation = p.readUserDebugParameter(self.target_orientation_id) * 2 * 3.14159  # 0 to 2pi
        except Exception as e:
            print(f"Error reading user debug parameter: {e}")
            target_orientation = 0  # Default to 0 if there's an error

        # Convert target velocity into 3d vector
        self.target_velocity = [magnitude * self.target_speed * np.cos(direction), magnitude * self.target_speed * np.sin(direction), 0]
        # Convert target orientation to quaternion [x, y, z, w]
        self.target_orientation = [0, 0, np.sin(target_orientation / 2), np.cos(target_orientation / 2)]
        
        # Debug: Print occasionally to verify values (every 100 steps)
        if self.steps_taken % 100 == 0:
            print(f"Step {self.steps_taken}: Target vel=[{self.target_velocity[0]:.3f}, {self.target_velocity[1]:.3f}], " +
                  f"Target orient angle={target_orientation:.2f} rad")

        # Clear previous debug lines
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id)
        self.debug_lines = []
        
        # Get current robot base position from PyBullet
        start_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        
        # Draw a line indicating the target velocity direction and magnitude
        # Length scales with velocity magnitude (multiply by a factor for visibility)
        velocity_scale = 2.0  # Make the line 2x longer than actual velocity for better visibility
        end_pos = [start_pos[0] + self.target_velocity[0] * velocity_scale, 
                   start_pos[1] + self.target_velocity[1] * velocity_scale, 
                   start_pos[2]]
        line_id = p.addUserDebugLine(start_pos, end_pos, [1, 0, 0], 3)  # Red line (thicker)
        self.debug_lines.append(line_id)
        
        # Draw a line indicating the target orientation direction (use the angle from the slider directly)
        # Fixed length for orientation (not scaled by velocity)
        orient_length = 0.5  # Fixed length in meters
        orient_end_pos = [start_pos[0] + orient_length * np.cos(target_orientation), 
                         start_pos[1] + orient_length * np.sin(target_orientation), 
                         start_pos[2]]
        orient_line_id = p.addUserDebugLine(start_pos, orient_end_pos, [0, 1, 0], 3)  # Green line (thicker)
        self.debug_lines.append(orient_line_id)

        # Get current velocity and orientation of the bot
        current_linear_vel, current_angular_vel = p.getBaseVelocity(self.robot_id)
        current_pos, current_orientation_quat = p.getBasePositionAndOrientation(self.robot_id)
        
        # Draw current velocity vector (dark red: [0.5, 0, 0])
        current_vel_end_pos = [current_pos[0] + current_linear_vel[0] * velocity_scale,
                               current_pos[1] + current_linear_vel[1] * velocity_scale,
                               current_pos[2]]
        current_vel_line_id = p.addUserDebugLine(current_pos, current_vel_end_pos, [0.5, 0, 0], 3)  # Dark red line
        self.debug_lines.append(current_vel_line_id)
        
        # Draw current orientation vector (dark green: [0, 0.5, 0])
        # Extract yaw angle from quaternion
        current_euler = p.getEulerFromQuaternion(current_orientation_quat)
        current_yaw = current_euler[2]  # z-axis rotation (yaw)
        current_orient_end_pos = [current_pos[0] + orient_length * np.cos(current_yaw),
                                  current_pos[1] + orient_length * np.sin(current_yaw),
                                  current_pos[2]]
        current_orient_line_id = p.addUserDebugLine(current_pos, current_orient_end_pos, [0, 0.5, 0], 3)  # Dark green line
        self.debug_lines.append(current_orient_line_id)

        return super().step(action)


if __name__ == "__main__":
    # To use a different robot, change the filename here

    urdf_file, save_path, save_prefix, model_path = utils.select_robot(load_model=True)

    min_z = get_min_z(urdf_file)
    # Create the environment. Stable-baselines will automatically call reset.
    env = VisualizationEnv(urdf_filename=urdf_file, start_position=[0, 0, -min_z])
    
    # Optionally wrap with video recording (comment out if you don't want videos)
    # env = wrappers.RecordVideo(env, video_folder='./videos/', name_prefix='servobot_demo', episode_trigger=lambda x: True)

    # check to make sure we didnt forget to import a model lmao
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train your model first using train.py.")
        exit(1)
    
    # Try to load the trained model
    try:
        model = PPO.load(model_path, env=env, device='cpu')
    except ValueError as e:
        if "Unexpected observation shape" in str(e):
            print(f"\n⚠️  ERROR: Model observation space mismatch!")
            print(f"The trained model expects a different observation space than the current environment.")
            print(f"This happens when you modify the environment after training.\n")
            print(f"SOLUTION: You need to retrain the model with the updated environment.")
            print(f"Run: python train.py\n")
            exit(1)
        else:
            raise


    print("Running trained model for 3 episodes...")
    for episode in range(3):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Use the trained model to predict the next action
            action, _states = model.predict(obs, deterministic=True)
            
            # Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
        print(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")

    env.close()
    print("Evaluation finished.")
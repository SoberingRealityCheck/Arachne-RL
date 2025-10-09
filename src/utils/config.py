''' 
Parameters for training are stored here! 
Also, local filepaths for different robot configurations.
All weights should be >= 0.0. Penalties are subtracted, not added.
'''

ROBOTS = {
    'simple_quadruped': {
        'urdf_file': "robots/simple_quadruped.urdf",
        'save_path': 'models/quadruped_checkpoints/',
        'save_prefix': 'quadruped_model',
        ### WEIGHTS FOR TRAINING ###
        #GOAL_APPROACH_WEIGHT': 5.0,
        #'GOAL_REACHED_BONUS': 200.0,  # Large bonus on touching the goal box
        'UPRIGHT_REWARD_WEIGHT': 0.5,
        'ACTION_PENALTY_WEIGHT': 0.01,
        'SHAKE_PENALTY_WEIGHT': 0.01,
        'SURVIVAL_WEIGHT': 0.02,  # Moderate incentive to stay alive
        'FALLEN_PENALTY': 20.0,  # Strong but not paralyzing penalty 
        'FORWARD_VEL_WEIGHT': 100.0,
        # New: discourage jumping/high vertical motion.
        'JUMP_PENALTY_WEIGHT': 0.1,     # Penalize excessive vertical velocity
        'HIGH_ALTITUDE_PENALTY_WEIGHT': 0.1,  # Penalize staying too high above ground """
        'ORIENTATION_REWARD_WEIGHT': 1.0,  # Reward for matching target orientation
        'ACTION_LIMIT': 0.2 # Proportional limit on joint angles. Should be between 0 and 1 # 0 is default and means no restriction. Otherwise smaller limit means more restriction.
    },


    'servobot': {
        'urdf_file': "robots/full_servobot/servobot.urdf",
        'save_path': 'models/servobot_checkpoints/',
        'save_prefix': 'servobot_model',
        ### WEIGHTS FOR TRAINING ###
        'UPRIGHT_REWARD_WEIGHT': 1.0,  # Increased - staying upright is critical
        'SHAKE_PENALTY_WEIGHT': 0.1,  # Increased - reduce jittery motion
        'SURVIVAL_WEIGHT': 0.02,  # Moderate incentive to prolong episodes
        'FALLEN_PENALTY': 30.0,  # Strong but not paralyzing penalty
        'FORWARD_VEL_WEIGHT': 10.0,  # REDUCED - was drowning out other signals (exp function handles scaling)
        # New: discourage jumping/high vertical motion.
        'JUMP_PENALTY_WEIGHT': 0.2,  # Increased - discourage jumping more
        'HIGH_ALTITUDE_PENALTY_WEIGHT': 0.2,  # Increased - keep robot grounded
        # New: Home Position Weight
        'HOME_POSITION_PENALTY_WEIGHT': 0.3,  # Reduced - allow more freedom of movement
        'TILT_PENALTY_WEIGHT': 0.1,  # Increased - penalize excessive tilting (pitch/roll)
        'ORIENTATION_REWARD_WEIGHT': 2.0,  # INCREASED - now properly scaled with exp function
        'ACTION_LIMIT': 0.2, # Proportional limit on joint angles. Should be between 0 and 1 # 0 is default and means no restriction. Otherwise smaller limit means more restriction.
        'INITIAL_MOMENTUM': 0.3,  # REDUCED - less chaotic starts help learning
    },
    'arachne': {
        'urdf_file': "robots/arachne/arachne.urdf",
        'save_path': 'models/arachne_checkpoints/',
        'save_prefix': 'arachne_model',
        'robot_name': 'arachne',
        ### WEIGHTS FOR TRAINING ###
        'UPRIGHT_REWARD_WEIGHT': 0.5,
        'SHAKE_PENALTY_WEIGHT': 0.01,
        'SURVIVAL_WEIGHT': 0.02,  # Moderate incentive to stay alive
        'FALLEN_PENALTY': 20.0,  # Strong but not paralyzing penalty   
        'FORWARD_VEL_WEIGHT': 10.0,  # Used in r_lin_vel (multiplied by exp, so max ~10)
        # New: discourage jumping/high vertical motion.
        'JUMP_PENALTY_WEIGHT': 0.1,     # Penalize excessive vertical velocity
        'HIGH_ALTITUDE_PENALTY_WEIGHT': 0.1,  # Penalize staying too high above ground
        'ORIENTATION_REWARD_WEIGHT': 1.0,  # Scaled by 10x in code (max ~10, balanced with velocity)
        'ACTION_LIMIT': 0.3,  # Increased from 0.5 to 0.3 for more precise control
        'INITIAL_MOMENTUM': 0.3,  # Reduced from 1.0 to help with stability during learning
        'TARGET_HEIGHT': 0.5,  # Target height for the arachne to maintain
        'TARGET_SPEED': 0.5,  # Reduced from 1.0 - start slower for better learning
    }
}

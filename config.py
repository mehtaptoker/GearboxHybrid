# --- Environment Configuration ---
MAX_GEARS = 15
MAX_STEPS_PER_EPISODE = 20
BOUNDARY_COMPLEXITY = 8
WORKSPACE_SIZE = 100.0  # The coordinate system will range from -50 to 50

# --- Physics Configuration ---
GEAR_MODULE = 1.0
MIN_TEETH = 8
MAX_TEETH = 40
MESHING_TOLERANCE = 1e-3  # Relative tolerance for checking meshing distance
GEAR_THICKNESS = 10.0  # Constant thickness for all gears (mm)
GEAR_DENSITY = 7.85    # Steel density (g/cm³)

# --- RL Training Hyperparameters ---
MODEL_ALGORITHM = "PPO"
POLICY = "MlpPolicy"
TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE = 0.0003
GAMMA = 0.99
TENSORBOARD_LOG_PATH = "./logs/"
BATCH_SIZE = 64
EPSILON = 0.2
# --- Reward Weights ---
W_RATIO_SUCCESS = 100.0
P_COLLISION = -10.0
WEIGHT_PENALTY = -0.01  # Penalty per gram of total gear mass
P_OUT_OF_BOUNDS = -10.0
P_NO_CONNECTION = -20.0  # Penalty if the final train doesn't connect I/O
P_EFFICIENCY = -0.1

# --- Environment Configuration ---
MAX_GEARS = 15
MAX_STEPS_PER_EPISODE = 20
BOUNDARY_COMPLEXITY = 8
WORKSPACE_SIZE = 100.0  # The coordinate system will range from -50 to 50
MAX_GAP_FILLING_ITERATIONS = 1000  # Maximum iterations for the gap filling algorithm

# --- Physics Configuration ---
GEAR_MODULE = 1.0
MIN_TEETH = 8
MAX_TEETH = 40
MIN_RADIUS = (MIN_TEETH * GEAR_MODULE) / 2.0  # Minimum radius for intermediate gears
MESHING_TOLERANCE = 1e-2  # Increased tolerance for more robust meshing checks
GEAR_THICKNESS = 10.0  # Constant thickness for all gears (mm)
GEAR_DENSITY = 7.85    # Steel density (g/cm³)
CONNECTION_PATH_WIDTH = 10.0  # Width of the connection path (mm)

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
P_OUT_OF_CONNECTION = -10.0  # Penalty for placing gear outside connection polygon
WEIGHT_PENALTY = -0.01  # Penalty per gram of total gear mass
P_OUT_OF_BOUNDS = -10.0
P_NO_CONNECTION = -20.0  # Penalty if the final train doesn't connect I/O
P_EFFICIENCY = -0.1
P_GEAR_COUNT_PENALTY = -0.1 # Penalty for each gear added to the system

# New parameters for composite reward function
ALPHA = 10.0  # Coefficient for torque ratio reward
BETA = 1.0    # Coefficient for connectivity penalty
COLLISION_PENALTY = -100.0  # Large penalty for collisions

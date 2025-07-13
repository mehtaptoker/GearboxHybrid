from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

import config
from environment import GearEnv
import os

def train():
    # Create a vectorized environment
    env = DummyVecEnv([lambda: Monitor(GearEnv())])

    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: Monitor(GearEnv())])
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best",
        log_path="logs/eval",
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    # Instantiate the PPO model
    model = PPO(
        policy=config.POLICY,
        env=env,
        verbose=1,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        tensorboard_log=config.TENSORBOARD_LOG_PATH,
        batch_size=config.BATCH_SIZE
    )

    # Create model directory
    os.makedirs("models", exist_ok=True)
    
    # Train the model with callbacks
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=eval_callback,
        tb_log_name="train",
        log_interval=10
    )

    # Save the final model
    model.save("models/final_model")
    print("Training complete. Model saved to models/final_model.zip")

if __name__ == "__main__":
    # Reduce timesteps for testing
    original_timesteps = config.TOTAL_TIMESTEPS
    config.TOTAL_TIMESTEPS = 10000  # Small dataset for testing
    
    train()
    
    # Restore original timesteps
    config.TOTAL_TIMESTEPS = original_timesteps

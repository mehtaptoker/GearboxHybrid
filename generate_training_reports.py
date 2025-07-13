import os
import config
from environment import GearEnv
from visualization import generate_report
from stable_baselines3 import PPO

def generate_model_reports(model_path="models/final_model", num_episodes=5):
    """Generate reports for model-generated gear systems."""
    # Load trained model
    model = PPO.load(model_path)
    
    # Create environment
    env = GearEnv()
    
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        # Run the episode
        while not done and not truncated:
            action, _ = model.predict(obs)
            obs, _, done, truncated, _ = env.step(action)
        
        # Generate report
        report_path = generate_report(env.state, "reports/trained")
        print(f"Generated trained model report {i+1}/{num_episodes} at {report_path}")

if __name__ == "__main__":
    generate_model_reports()

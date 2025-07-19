import torch
import numpy as np
from environment import GearEnv
from train_torch import PolicyNetwork
import json
import os
from visualization import generate_report

def evaluate_model(model_path, data_dir, report_dir="reports/evaluation", num_episodes=10):
    """
    Evaluate a trained model on multiple episodes and generate reports.
    
    Args:
        model_path (str): Path to the trained policy model
        data_dir (str): Directory containing training data
        num_episodes (int): Number of evaluation episodes
    
    Returns:
        dict: Evaluation results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = GearEnv()
    
    # Load policy
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0])
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    
    results = []
    # Use the provided report directory
    os.makedirs(report_dir, exist_ok=True)
    
    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = policy(state_tensor)
                action = mu.squeeze(0).cpu().numpy()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
        
        # Generate report for this episode
        report_path = generate_report(env.state, os.path.join(report_dir, f"episode_{i+1}"))
        
        results.append({
            "episode": i+1,
            "reward": episode_reward,
            "steps": episode_steps,
            "report_path": report_path,
            "gear_count": len(env.state.gears),
            "target_ratio": env.state.target_ratio,
            "achieved_ratio": env.state.calculate_ratio()
        })
    
    # Save overall results
    with open(os.path.join(report_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    import argparse
    import config
    parser = argparse.ArgumentParser(description='Evaluate a trained model.')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data-dir', type=str, default='data/intermediate', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='reports/evaluation', help='Output directory for reports')
    parser.add_argument('--num-episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--max-gears', type=int, default=config.MAX_GEARS, help='Maximum number of gears')
    args = parser.parse_args()
    
    # Set max gears from command line
    config.MAX_GEARS = args.max_gears
    
    results = evaluate_model(args.model, args.data_dir, args.output_dir, args.num_episodes)
    print(f"Evaluation complete. Results saved to reports/evaluation/evaluation_results.json")
    print(f"Average reward: {np.mean([r['reward'] for r in results]):.2f}")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import argparse
import os

import config
from environment import GearEnv

class PolicyNetwork(nn.Module):
    """Policy network for PPO"""
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.log_std = nn.Parameter(torch.zeros(1, output_size))
        
    def forward(self, x):
        mu = self.fc(x)
        std = self.log_std.exp().expand_as(mu)
        return mu, std

class ValueNetwork(nn.Module):
    """Value network for PPO"""
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.fc(x)
#boundary constraint, torque and the mass --> linear combination
class PPOAgent:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.obs_size = self.env.observation_space.shape[0]
        self.action_size = 5  # New action space: [action_type, x, y, num_teeth, z_layer]
        
        # Initialize networks and move to device
        self.policy = PolicyNetwork(self.obs_size, self.action_size).to(device)
        self.value_net = ValueNetwork(self.obs_size).to(device)
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=config.LEARNING_RATE)
        
        # Memory buffer
        self.memory = deque(maxlen=10000)
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        mu, std = self.policy(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action.squeeze(0).cpu().detach().numpy(), log_prob.cpu().detach().numpy()
    
    def store_transition(self, state, action, log_prob, reward, next_state, done):
        self.memory.append((state, action, log_prob, reward, next_state, done))
    
    def update(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
            
        # Sample batch from memory
        batch = random.sample(self.memory, config.BATCH_SIZE)
        states, actions, old_log_probs, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to device
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Calculate advantages
        values = self.value_net(states)
        next_values = self.value_net(next_states)
        advantages = rewards + config.GAMMA * next_values * (1 - dones) - values
        
        # Normalize advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        advantages = advantages.detach()
        
        # Calculate new log probabilities
        mu, std = self.policy(states)
        dist = torch.distributions.Normal(mu, std)
        new_log_probs = dist.log_prob(actions).sum(1, keepdim=True)
        
        # PPO loss
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-config.EPSILON, 1+config.EPSILON) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values, rewards + config.GAMMA * next_values * (1 - dones))
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def train(self, total_timesteps):
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0

        # Main training loop over the total number of steps
        for t in range(1, total_timesteps + 1):
            # 1. Select a raw, continuous action from the policy network
            raw_action, log_prob = self.select_action(state)
            
            # 2. Process the raw action into a valid, constrained action for the environment
            processed_action = np.zeros_like(raw_action)
            
            # Action type: Use argmax for a discrete choice (e.g., 0 for add_gear, 1 for connect)
            processed_action[0] = np.argmax(raw_action[0]) 

            # X and Y position: Use tanh to squash the value between [-1, 1], then scale to the workspace size
            processed_action[1] = np.tanh(raw_action[1]) * config.WORKSPACE_SIZE / 2
            processed_action[2] = np.tanh(raw_action[2]) * config.WORKSPACE_SIZE / 2
            
            # Number of teeth: Ensure the value is a positive integer within a valid range
            num_teeth_raw = int(np.round(abs(raw_action[3])))
            processed_action[3] = np.clip(num_teeth_raw, config.MIN_TEETH, config.MAX_TEETH) # e.g., 12 to 60
            
            # Z layer: Round to the nearest integer
            processed_action[4] = int(np.round(raw_action[4]))
            
            # 3. Take a step in the environment using the PROCESSED action
            next_state, reward, terminated, truncated, _ = self.env.step(processed_action)
            done = terminated or truncated
            
            # 4. Store the RAW action and its log_prob, which PPO needs for the update
            self.store_transition(state, raw_action, log_prob, reward, next_state, done)
            
            # Update the current state and track episode statistics
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # 5. Check if the episode has finished (CORRECTLY INDENTED)
            if done:
                # Log the results of the completed episode
                gear_count = len(self.env.state.gears) if self.env.state else 0
                print(f"Step: {t}, Episode Reward: {episode_reward:.2f}, Length: {episode_length}, Gears: {gear_count}")
                
                # Update the agent's networks using the data from the episode
                if len(self.memory) >= config.BATCH_SIZE:
                    self.update()
                
                # Reset the environment for the next episode
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0

        # After the entire training loop is finished, save the final models
        torch.save(self.policy.state_dict(), "gear_generator_policy.pth")
        torch.save(self.value_net.state_dict(), "gear_generator_value.pth")
        print("Training complete. Models saved.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train gear generation model')
    parser.add_argument('--gpu', type=int, default=None, 
                        help='GPU ID to use (default: None uses CPU)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing training data. If not provided, new data will be generated.')
    parser.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE,
                        help='Learning rate for optimizer')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--total-timesteps', type=int, default=config.TOTAL_TIMESTEPS,
                        help='Total timesteps for training')
    parser.add_argument('--gamma', type=float, default=config.GAMMA,
                        help='Discount factor gamma')
    parser.add_argument('--epsilon', type=float, default=config.EPSILON,
                        help='PPO clipping epsilon')
    parser.add_argument('--max-gears', type=int, default=config.MAX_GEARS,
                        help='Maximum gears per episode')
    parser.add_argument('--max-steps', type=int, default=config.MAX_STEPS_PER_EPISODE,
                        help='Maximum steps per episode')
    parser.add_argument('--min-teeth', type=int, default=config.MIN_TEETH,
                        help='Minimum number of teeth per gear')
    parser.add_argument('--max-teeth', type=int, default=config.MAX_TEETH,
                        help='Maximum number of teeth per gear')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level (0: none, 1: basic, 2: detailed)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to a pre-trained model to continue training from')
    
    args = parser.parse_args()
    
    # Set config parameters from command line arguments
    config.DATA_DIR = args.data_dir
    config.LEARNING_RATE = args.learning_rate
    config.BATCH_SIZE = args.batch_size
    config.TOTAL_TIMESTEPS = args.total_timesteps
    config.GAMMA = args.gamma
    config.EPSILON = args.epsilon
    config.MAX_GEARS = args.max_gears
    config.MAX_STEPS_PER_EPISODE = args.max_steps
    config.MIN_TEETH = args.min_teeth
    config.MAX_TEETH = args.max_teeth
    
    # Set device (GPU if available)
    device = torch.device('cpu')
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        print("Using CPU")
    
    env = GearEnv(data_dir=config.DATA_DIR, verbose=args.verbose)
    agent = PPOAgent(env, device)

    # Load pre-trained model if provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading pre-trained model from {args.model_path}")
        agent.policy.load_state_dict(torch.load(args.model_path, map_location=device))
        # If you also save the value network, load it here
        # agent.value_net.load_state_dict(torch.load(value_model_path, map_location=device))
    else:
        print("No pre-trained model found, starting from scratch.")

    agent.train(config.TOTAL_TIMESTEPS)

if __name__ == "__main__":
    main()

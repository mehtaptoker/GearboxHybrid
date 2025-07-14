import time
import os
import matplotlib.pyplot as plt
from datetime import datetime

def monitor_training(log_dir="logs", report_interval=60):
    """
    Monitor training progress and generate periodic reports.
    
    Args:
        log_dir (str): Directory containing training logs
        report_interval (int): Time interval between reports in seconds
    """
    # Create monitoring directory
    monitor_dir = os.path.join(log_dir, "monitoring")
    os.makedirs(monitor_dir, exist_ok=True)
    
    print(f"Monitoring training progress. Reports will be generated every {report_interval} seconds.")
    
    episode_rewards = []
    episode_lengths = []
    timestamps = []
    
    try:
        while True:
            # Check for latest log file
            log_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
            if not log_files:
                time.sleep(5)
                continue
                
            latest_log = max(log_files, key=lambda f: os.path.getmtime(os.path.join(log_dir, f)))
            log_path = os.path.join(log_dir, latest_log)
            
            # Parse log file (simplified example)
            # In a real implementation, you would use TensorBoard or similar
            with open(log_path, "r") as f:
                lines = f.readlines()
                for line in lines[-100:]:  # Check last 100 lines
                    if "Episode Reward" in line:
                        parts = line.split("Episode Reward: ")[1].split(",")
                        reward = float(parts[0])
                        length = int(parts[1].split(": ")[1])
                        
                        episode_rewards.append(reward)
                        episode_lengths.append(length)
                        timestamps.append(datetime.now())
            
            # Generate report if we have data
            if episode_rewards:
                # Create plot
                plt.figure(figsize=(12, 6))
                
                # Reward plot
                plt.subplot(1, 2, 1)
                plt.plot(episode_rewards, 'b-')
                plt.title("Episode Rewards")
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                
                # Length plot
                plt.subplot(1, 2, 2)
                plt.plot(episode_lengths, 'g-')
                plt.title("Episode Lengths")
                plt.xlabel("Episode")
                plt.ylabel("Steps")
                
                # Save report
                report_path = os.path.join(monitor_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.tight_layout()
                plt.savefig(report_path)
                plt.close()
                
                print(f"Generated monitoring report: {report_path}")
            
            time.sleep(report_interval)
            
    except KeyboardInterrupt:
        print("Monitoring stopped.")

if __name__ == "__main__":
    monitor_training()

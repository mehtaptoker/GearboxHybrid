# Generative Design of 2D Mechanical Gear Systems

## Comprehensive Documentation
This repository contains detailed documentation covering all aspects of the system:

1. [System Overview](docs/overview.md) - High-level architecture and workflow
2. [Physics and Constraints](docs/physics.md) - Gear representation and physical constraints
3. [Reinforcement Learning](docs/rl_training.md) - Training process and agent implementation

## Getting Started
### Installation
```bash
git clone https://github.com/mingwucn/GearGen.git
cd GearGen
pip install -r requirements.txt
```

### Workflow
1. **Training**:
   ```bash
   ./run_train.sh [options]
   ```
   Example with GPU:
   ```bash
   ./run_train.sh --gpu 0 --data-dir data/intermediate
   ```

2. **Testing**:
   ```bash
   ./test.sh
   ```

3. **Evaluation**:
   ```bash
   ./run_eval.sh --model models/best_policy.pth
   ```

### Script Options
- `run_train.sh`: Supports all parameters from train_torch.py
- `test.sh`: Runs demo cases with visualization
- `run_eval.sh`: Generates evaluation reports with metrics

## Research Foundations
This implementation is based on rigorous research in:
- Generative mechanical design
- Topology optimization
- Reinforcement learning for constrained placement problems
- Physics-based simulation for mechanical systems

## Validation Methodology
We use four standardized test cases:
1. **Simple Reduction**: 2:1 reduction with two gears
2. **Multi-stage Reduction**: 10:1 reduction requiring idler gears
3. **Constrained Reversal**: 1:1 reversal in L-shaped boundary
4. **Layered Solution**: Systems requiring different z-layers for shaft crossings

## Gear Generation Process
Our system uses a two-step approach for generating gears:

1. **Geometric Placement**: 
   - Determines reference diameter and positions
   - Ensures gears fit within boundaries and avoid collisions
   - Uses reinforcement learning to optimize placement

2. **Teeth Assignment**:
   - Assigns teeth count based on desired gear ratios
   - Ensures meshing compatibility between gears
   - Optimizes for efficiency and manufacturability

This decoupled approach allows for more efficient exploration of valid gear configurations.

## Explainability
The system provides detailed reward contribution reports:
```
Final Ratio Reward: +100.0
Collision Penalty: -0.0
Boundary Penalty: -0.0
Efficiency Penalty (3 gears): -5.0
Total Score: +95.0
```

## File Structure
```text
GearGen/
├── components.py       # Data structures
├── environment.py      # RL environment
├── physics.py          # Physics calculations
├── train_torch.py      # Training implementation
├── evaluation.py       # Evaluation tools
├── visualization.py    # Visualization tools
├── run_train.sh        # Training script
├── test.sh             # Testing script
└── run_eval.sh         # Evaluation script
├── data/               # Sample data and constraints
├── docs/               # Documentation
├── logs/               # Training logs
├── models/             # Trained models
└── reports/            # Evaluation reports
```

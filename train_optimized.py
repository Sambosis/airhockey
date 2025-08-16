#!/usr/bin/env python3
"""
Optimized Air Hockey Training Script

This script runs the Air Hockey DQN training with optimized parameters
for faster convergence and better performance.

Usage:
    python train_optimized.py [additional_args...]
    
Example:
    python train_optimized.py --episodes 5000 --device cuda
"""

import sys
import subprocess
from optimized_config import get_optimized_args, print_optimization_summary


def main():
    """Run optimized training with optional additional arguments."""
    
    # Print optimization summary
    print_optimization_summary()
    
    # Get optimized arguments
    optimized_args = get_optimized_args()
    
    # Add any additional arguments passed to this script
    additional_args = sys.argv[1:]
    
    # Combine arguments
    all_args = ["python3", "main.py"] + optimized_args + additional_args
    
    print("\n" + "="*60)
    print("STARTING OPTIMIZED TRAINING")
    print("="*60)
    print("Command:", " ".join(all_args))
    print("="*60 + "\n")
    
    # Run the training
    try:
        subprocess.run(all_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
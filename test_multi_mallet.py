#!/usr/bin/env python3
"""
Test script to demonstrate the multi-mallet air hockey functionality.
"""

import sys
from src.config import Config
from src.env import AirHockeyEnv
import numpy as np

def test_multi_mallet(num_mallets=2):
    """Test the environment with multiple mallets per side."""
    print(f"\n{'='*60}")
    print(f"Testing with {num_mallets} mallet(s) per side")
    print(f"{'='*60}")
    
    # Create config with specified number of mallets
    config = Config(
        num_mallets_per_side=num_mallets,
        episodes=1,
        visualize_every_n=0,
        checkpoint_every=0,
    )
    
    # Create environment
    env = AirHockeyEnv(config)
    
    # Reset and get initial observations
    obs_left, obs_right = env.reset()
    
    print(f"Environment created successfully!")
    print(f"- Table dimensions: {config.width} x {config.height}")
    print(f"- Mallets per side: {config.num_mallets_per_side}")
    print(f"- Observation dimension: {len(obs_left)} (expected: {config.obs_dim})")
    print(f"- Action space: 5 actions (stay, up, down, left, right)")
    
    # Print mallet positions
    print(f"\nInitial mallet positions:")
    print(f"  Left mallets:")
    for i, mallet in enumerate(env.left_mallets):
        print(f"    Mallet {i+1}: x={mallet.x:.1f}, y={mallet.y:.1f}")
    print(f"  Right mallets:")
    for i, mallet in enumerate(env.right_mallets):
        print(f"    Mallet {i+1}: x={mallet.x:.1f}, y={mallet.y:.1f}")
    
    # Run a few steps with random actions
    print(f"\nRunning 10 steps with random actions...")
    for step in range(10):
        # Random actions for both sides
        action_left = np.random.randint(0, 5)
        action_right = np.random.randint(0, 5)
        
        # Step the environment
        obs_left, obs_right, r_left, r_right, done, info = env.step(action_left, action_right)
        
        if done:
            print(f"  Step {step+1}: Episode ended! Goal: {info.get('goal', 'none')}")
            break
        else:
            print(f"  Step {step+1}: Rewards - Left: {r_left:+.3f}, Right: {r_right:+.3f}")
    
    print(f"\nTest completed successfully!")
    return True

def main():
    """Run tests with different numbers of mallets."""
    print("Multi-Mallet Air Hockey Test Suite")
    print("=" * 60)
    
    # Test with different numbers of mallets
    for num_mallets in [1, 2, 3, 4]:
        try:
            success = test_multi_mallet(num_mallets)
            if not success:
                print(f"❌ Test failed for {num_mallets} mallet(s)")
                return 1
        except Exception as e:
            print(f"❌ Error testing {num_mallets} mallet(s): {e}")
            return 1
    
    print("\n" + "=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
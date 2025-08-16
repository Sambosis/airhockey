#!/usr/bin/env python3
"""
Benchmarking script for Air Hockey DQN training optimization.
Compares different configurations and measures performance improvements.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.env import AirHockeyEnv
from src.agents.dqn import DQNAgent
from src.agents.dueling_dqn import DuelingDQNAgent
from src.render import Renderer
from src.train import Trainer
from src.utils import seed_everything


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single configuration."""
    config_name: str
    episodes_trained: int
    total_time_seconds: float
    final_avg_reward_left: float
    final_avg_reward_right: float
    final_epsilon: float
    total_env_steps: int
    avg_episode_length: float
    win_rate_left: float
    win_rate_right: float
    steps_per_second: float
    memory_usage_mb: float
    config_params: Dict[str, Any]


class Benchmarker:
    """Benchmark different training configurations."""
    
    def __init__(self, base_config: Config, output_dir: str = "benchmark_results"):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
    def run_configuration(
        self, 
        config_name: str, 
        config_overrides: Dict[str, Any],
        episodes: int = 100,
        render: bool = False
    ) -> BenchmarkResult:
        """Run a single configuration and collect metrics."""
        
        print(f"\n{'='*60}")
        print(f"Running benchmark: {config_name}")
        print(f"{'='*60}")
        
        # Create config with overrides
        config = Config()
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Set benchmark-specific settings
        config.episodes = episodes
        config.visualize_every_n = 0 if not render else 50
        config.checkpoint_every = episodes + 1  # Don't checkpoint during benchmark
        
        # Seed for reproducibility
        seed_everything(config.seed)
        
        # Create environment
        env = AirHockeyEnv(config)
        
        # Create agents based on algorithm
        if config.algo == "dqn":
            agent_class = DQNAgent
        else:
            agent_class = DuelingDQNAgent
            
        agent_left = agent_class(
            obs_dim=config.obs_dim,
            action_space_n=config.action_space_n,
            lr=config.lr,
            gamma=config.gamma,
            batch_size=config.batch_size,
            eps_start=config.eps_start,
            eps_end=config.eps_end,
            eps_decay_frames=config.eps_decay_frames,
            target_sync=config.target_sync,
            learn_start=config.learn_start,
            buffer_capacity=config.buffer_capacity,
            device=config.device_str,
            seed=config.seed,
        )
        
        agent_right = agent_class(
            obs_dim=config.obs_dim,
            action_space_n=config.action_space_n,
            lr=config.lr,
            gamma=config.gamma,
            batch_size=config.batch_size,
            eps_start=config.eps_start,
            eps_end=config.eps_end,
            eps_decay_frames=config.eps_decay_frames,
            target_sync=config.target_sync,
            learn_start=config.learn_start,
            buffer_capacity=config.buffer_capacity,
            device=config.device_str,
            seed=config.seed + 1,
        )
        
        # Create renderer
        renderer = Renderer(
            width=config.width,
            height=config.height,
            fps=config.render_fps,
            headless=not render
        )
        
        # Create trainer
        trainer = Trainer(
            env=env,
            agent_left=agent_left,
            agent_right=agent_right,
            renderer=renderer,
            visualize_every_n=config.visualize_every_n,
            device=config.device_str,
            checkpoint_dir=config.checkpoint_dir,
            checkpoint_every=config.checkpoint_every,
            seed=config.seed,
        )
        
        # Measure training time
        start_time = time.time()
        
        # Run training
        trainer.run(episodes)
        
        # Calculate metrics
        total_time = time.time() - start_time
        
        # Get final statistics
        final_avg_reward_left = float(np.mean(trainer.recent_returns_left)) if trainer.recent_returns_left else 0.0
        final_avg_reward_right = float(np.mean(trainer.recent_returns_right)) if trainer.recent_returns_right else 0.0
        avg_episode_length = float(np.mean(trainer.recent_lengths)) if trainer.recent_lengths else 0.0
        
        # Calculate win rates (simplified - based on positive rewards)
        wins_left = sum(1 for r in trainer.recent_returns_left if r > 0)
        wins_right = sum(1 for r in trainer.recent_returns_right if r > 0)
        total_recent = len(trainer.recent_returns_left)
        
        win_rate_left = wins_left / total_recent if total_recent > 0 else 0.0
        win_rate_right = wins_right / total_recent if total_recent > 0 else 0.0
        
        # Memory usage (approximate)
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
        else:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Create result
        result = BenchmarkResult(
            config_name=config_name,
            episodes_trained=episodes,
            total_time_seconds=total_time,
            final_avg_reward_left=final_avg_reward_left,
            final_avg_reward_right=final_avg_reward_right,
            final_epsilon=agent_left.epsilon,
            total_env_steps=trainer.total_env_steps,
            avg_episode_length=avg_episode_length,
            win_rate_left=win_rate_left,
            win_rate_right=win_rate_right,
            steps_per_second=trainer.total_env_steps / total_time if total_time > 0 else 0,
            memory_usage_mb=memory_mb,
            config_params=config_overrides
        )
        
        self.results.append(result)
        
        # Clean up
        renderer.close()
        del trainer, agent_left, agent_right, env, renderer
        
        return result
    
    def compare_configurations(self, configurations: Dict[str, Dict[str, Any]], episodes: int = 100):
        """Run multiple configurations and compare results."""
        
        print("\n" + "="*80)
        print("STARTING BENCHMARK COMPARISON")
        print("="*80)
        
        for config_name, config_overrides in configurations.items():
            result = self.run_configuration(config_name, config_overrides, episodes)
            self.print_result(result)
        
        # Save results
        self.save_results()
        
        # Print comparison
        self.print_comparison()
    
    def print_result(self, result: BenchmarkResult):
        """Print a single benchmark result."""
        print(f"\nResults for {result.config_name}:")
        print(f"  Episodes trained: {result.episodes_trained}")
        print(f"  Total time: {result.total_time_seconds:.2f}s")
        print(f"  Steps/second: {result.steps_per_second:.2f}")
        print(f"  Avg episode length: {result.avg_episode_length:.1f}")
        print(f"  Final avg reward (L/R): {result.final_avg_reward_left:.3f} / {result.final_avg_reward_right:.3f}")
        print(f"  Win rate (L/R): {result.win_rate_left:.2%} / {result.win_rate_right:.2%}")
        print(f"  Memory usage: {result.memory_usage_mb:.2f} MB")
    
    def print_comparison(self):
        """Print comparison table of all results."""
        if not self.results:
            return
        
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON")
        print("="*80)
        
        # Sort by average reward
        sorted_results = sorted(
            self.results, 
            key=lambda r: (r.final_avg_reward_left + r.final_avg_reward_right) / 2,
            reverse=True
        )
        
        print(f"\n{'Config':<25} {'Avg Reward':<15} {'Win Rate':<15} {'Steps/s':<12} {'Time (s)':<10}")
        print("-" * 80)
        
        for result in sorted_results:
            avg_reward = (result.final_avg_reward_left + result.final_avg_reward_right) / 2
            avg_win_rate = (result.win_rate_left + result.win_rate_right) / 2
            
            print(f"{result.config_name:<25} {avg_reward:<15.3f} {avg_win_rate:<15.2%} "
                  f"{result.steps_per_second:<12.2f} {result.total_time_seconds:<10.2f}")
        
        # Calculate improvement
        if len(sorted_results) >= 2:
            best = sorted_results[0]
            baseline = sorted_results[-1]
            
            avg_reward_best = (best.final_avg_reward_left + best.final_avg_reward_right) / 2
            avg_reward_baseline = (baseline.final_avg_reward_left + baseline.final_avg_reward_right) / 2
            
            if avg_reward_baseline != 0:
                improvement = ((avg_reward_best - avg_reward_baseline) / abs(avg_reward_baseline)) * 100
                print(f"\nBest configuration '{best.config_name}' shows {improvement:.1f}% improvement over baseline")
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_{timestamp}.json"
        
        results_dict = [asdict(r) for r in self.results]
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    """Run benchmarks with different configurations."""
    parser = argparse.ArgumentParser(description="Benchmark Air Hockey DQN training")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per configuration")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer episodes")
    args = parser.parse_args()
    
    # Base configuration
    base_config = Config()
    
    # Create benchmarker
    benchmarker = Benchmarker(base_config)
    
    # Define configurations to test
    configurations = {
        "Baseline (Original)": {
            "algo": "dqn",
            "lr": 1e-4,
            "batch_size": 1024,
            "target_sync": 50000,
            "eps_decay_frames": 3000000,
        },
        "Optimized DQN": {
            "algo": "dqn",
            "lr": 5e-4,
            "batch_size": 256,
            "target_sync": 10000,
            "eps_decay_frames": 1000000,
        },
        "Optimized Dueling": {
            "algo": "dueling",
            "lr": 5e-4,
            "batch_size": 256,
            "target_sync": 10000,
            "eps_decay_frames": 1000000,
        },
        "High Learning Rate": {
            "algo": "dueling",
            "lr": 1e-3,
            "batch_size": 256,
            "target_sync": 5000,
            "eps_decay_frames": 500000,
        },
    }
    
    if args.quick:
        # Quick test with fewer episodes
        episodes = 20
        configurations = {
            "Baseline": configurations["Baseline (Original)"],
            "Optimized": configurations["Optimized Dueling"],
        }
    else:
        episodes = args.episodes
    
    # Run benchmarks
    benchmarker.compare_configurations(configurations, episodes)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
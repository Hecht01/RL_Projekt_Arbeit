#!/usr/bin/env python3
"""
Main script for Task 1: Reinforcement Learning in Gridworld Environments

This script:
1. Solves the MazeWater2 environment using optimized hyperparameters
2. Creates a custom challenging environment (SpiralMaze)
3. Plots results focusing on convergence episodes
4. Compares different RL algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import random
from collections import defaultdict

# Import our modules
from gridworld import MazeWater2, GridworldEnv
from Agents import RLAgentFactory, compare_agents, plot_comparison, print_performance_summary
from optimizer import OptunaOptimizer, optimize_multiple_agents, print_optimization_summary
from plot import plot_q_table, plot_v_table


class SpiralMaze(GridworldEnv):
    """
    Custom challenging environment: A spiral maze with multiple dead ends,
    narrow passages, and strategic placement of water traps.

    This environment is designed to challenge RL agents with:
    - Long optimal path requiring strategic planning
    - Multiple dead ends that can trap greedy policies
    - Narrow passages that require precise navigation
    - Water traps that punish exploration mistakes
    """

    def __init__(self):
        self.step_cost = -1
        self.g_reward = 100
        self.o_reward = -50

        # Create a 15x15 spiral maze with strategic water placement
        self.grid = [
            ['S', ' ', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
            [' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
            ['#', ' ', '#', ' ', '#', '#', '#', '#', '#', '#', '#', '#', '#', ' ', '#'],
            ['#', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#'],
            ['#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#'],
            ['#', ' ', '#', ' ', '#', ' ', '#', 'O', 'O', 'O', '#', ' ', '#', ' ', '#'],
            ['#', ' ', '#', ' ', '#', ' ', '#', 'O', 'G', 'O', '#', ' ', '#', ' ', '#'],
            ['#', ' ', '#', ' ', '#', ' ', '#', 'O', 'O', 'O', '#', ' ', '#', ' ', '#'],
            ['#', ' ', '#', ' ', '#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', '#'],
            ['#', ' ', '#', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#'],
            ['#', ' ', '#', ' ', '#', '#', '#', '#', '#', '#', '#', '#', '#', ' ', '#'],
            ['#', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
            ['#', ' ', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
            ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', ' ']
        ]

        GridworldEnv.__init__(self)


def optimize_for_mazewater2(num_trials: int = 50) -> Dict:
    """
    Optimize hyperparameters specifically for MazeWater2 environment
    """
    print("=" * 60)
    print("OPTIMIZING HYPERPARAMETERS FOR MAZEWATER2")
    print("=" * 60)

    env = MazeWater2()

    # Define agents to optimize
    agent_types = ['qlearning', 'sarsa', 'expected_sarsa', 'double_qlearning']

    results = optimize_multiple_agents(
        env=env,
        agent_types=agent_types,
        n_trials=num_trials,
        num_runs=5,
        episodes=600,
        optimization_metric='combined'
    )

    print_optimization_summary(results)
    return results


def solve_mazewater2_with_best_params(optimization_results: Dict) -> Dict:
    """
    Solve MazeWater2 using the best parameters found through optimization
    """
    print("\n" + "=" * 60)
    print("SOLVING MAZEWATER2 WITH OPTIMIZED PARAMETERS")
    print("=" * 60)

    env = MazeWater2()

    # Get the best agent type and parameters
    best_agent_type = max(optimization_results.keys(),
                          key=lambda x: optimization_results[x]['best_score'])
    best_params = optimization_results[best_agent_type]['best_params']

    print(f"Using best agent: {best_agent_type}")
    print(f"Best parameters: {best_params}")

    # Train multiple agents for comparison
    agent_types = [best_agent_type, 'qlearning', 'sarsa', 'expected_sarsa']

    results = {}
    for agent_type in agent_types:
        print(f"\nTraining {agent_type}...")

        # Use optimized params for best agent, default for others
        if agent_type == best_agent_type:
            kwargs = best_params
        else:
            kwargs = {'alpha': 0.1, 'gamma': 0.95, 'epsilon': 0.1}

        agent_results = []
        for run in range(10):
            print(f"  Run {run + 1}/10")
            agent = RLAgentFactory.create_agent(agent_type, env, **kwargs)
            rewards = agent.train(episodes=800, verbose=False)
            agent_results.append(rewards)

        results[agent_type] = {
            'rewards': np.array(agent_results),
            'mean_rewards': np.mean(agent_results, axis=0),
            'std_rewards': np.std(agent_results, axis=0),
            'final_performance': np.mean([r[-50:] for r in agent_results]),
            'convergence_episode': find_convergence_episode(agent_results),
            'params': kwargs if agent_type == best_agent_type else 'default'
        }

    return results


def solve_custom_environment() -> Dict:
    """
    Solve the custom SpiralMaze environment
    """
    print("\n" + "=" * 60)
    print("SOLVING CUSTOM SPIRAL MAZE ENVIRONMENT")
    print("=" * 60)

    env = SpiralMaze()

    # First, let's quickly optimize for this environment
    print("Quick optimization for SpiralMaze...")
    optimizer = OptunaOptimizer(
        env=env,
        agent_type='qlearning',
        num_runs=3,
        episodes=1000,
        optimization_metric='combined'
    )

    opt_result = optimizer.optimize(n_trials=30)
    best_params = opt_result['best_params']

    print(f"Best parameters for SpiralMaze: {best_params}")

    # Train agents with optimized parameters
    agent_types = ['qlearning', 'sarsa', 'expected_sarsa']
    results = {}

    for agent_type in agent_types:
        print(f"\nTraining {agent_type} on SpiralMaze...")

        # Use optimized params for Q-learning, default for others
        if agent_type == 'qlearning':
            kwargs = best_params
        else:
            kwargs = {'alpha': 0.1, 'gamma': 0.95, 'epsilon': 0.15}

        agent_results = []
        for run in range(5):
            print(f"  Run {run + 1}/5")
            agent = RLAgentFactory.create_agent(agent_type, env, **kwargs)
            rewards = agent.train(episodes=1200, verbose=False)
            agent_results.append(rewards)

        results[agent_type] = {
            'rewards': np.array(agent_results),
            'mean_rewards': np.mean(agent_results, axis=0),
            'std_rewards': np.std(agent_results, axis=0),
            'final_performance': np.mean([r[-50:] for r in agent_results]),
            'convergence_episode': find_convergence_episode(agent_results)
        }

    return results


def find_convergence_episode(agent_results: List[List[float]], threshold: float = 0.85) -> int:
    """Find the episode where the agent converges to a good solution"""
    mean_rewards = np.mean(agent_results, axis=0)

    # Use a moving average to smooth the data
    window_size = 25
    if len(mean_rewards) < window_size:
        return len(mean_rewards) - 1

    moving_avg = np.convolve(mean_rewards, np.ones(window_size) / window_size, mode='valid')
    max_reward = np.max(moving_avg)
    target_reward = max_reward * threshold

    for i, reward in enumerate(moving_avg):
        if reward >= target_reward:
            return i + window_size - 1

    return len(mean_rewards) - 1


def plot_convergence_focused(results: Dict, env_name: str, focus_window: int = 150):
    """
    Plot results focusing on convergence episodes as specified in the task
    """
    plt.figure(figsize=(14, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, (agent_type, data) in enumerate(results.items()):
        mean_rewards = data['mean_rewards']
        std_rewards = data['std_rewards']
        convergence_episode = data['convergence_episode']

        # Focus on episodes around convergence
        start_episode = max(0, convergence_episode - focus_window // 2)
        end_episode = min(len(mean_rewards), convergence_episode + focus_window)

        episodes = range(start_episode, end_episode)
        focused_mean = mean_rewards[start_episode:end_episode]
        focused_std = std_rewards[start_episode:end_episode]

        color = colors[i % len(colors)]

        plt.plot(episodes, focused_mean,
                 label=f"{agent_type} (conv: {convergence_episode}, final: {data['final_performance']:.1f})",
                 color=color, linewidth=2)

        plt.fill_between(episodes,
                         focused_mean - focused_std,
                         focused_mean + focused_std,
                         alpha=0.2, color=color)

        # Mark convergence point
        plt.axvline(x=convergence_episode, color=color, linestyle='--', alpha=0.7)

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.title(f'{env_name} - Convergence Analysis (Focused View)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_full_training(results: Dict, env_name: str):
    """
    Plot full training curves for comparison
    """
    plt.figure(figsize=(14, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, (agent_type, data) in enumerate(results.items()):
        mean_rewards = data['mean_rewards']
        std_rewards = data['std_rewards']

        color = colors[i % len(colors)]

        plt.plot(mean_rewards,
                 label=f"{agent_type} (final: {data['final_performance']:.1f})",
                 color=color, linewidth=2)

        plt.fill_between(range(len(mean_rewards)),
                         mean_rewards - std_rewards,
                         mean_rewards + std_rewards,
                         alpha=0.2, color=color)

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.title(f'{env_name} - Full Training Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_learned_policy(env, agent_type: str, params: Dict):
    """
    Visualize the learned policy using the plotting functions
    """
    print(f"\nVisualizing learned policy for {agent_type}...")

    # Train one agent to get the final Q-table
    agent = RLAgentFactory.create_agent(agent_type, env, **params)
    agent.train(episodes=1000, verbose=False)

    # Get policy and plot
    policy = agent.get_policy()

    # Convert policy to probability distribution for plotting
    policy_probs = np.zeros((env.num_states(), env.num_actions()))
    for state in range(env.num_states()):
        policy_probs[state, policy[state]] = 1.0

    # Plot Q-table and policy
    plot_q_table(env, agent.Q, policy_probs)


def print_detailed_results(results: Dict, env_name: str):
    """
    Print detailed results for the environment
    """
    print(f"\n{'=' * 60}")
    print(f"DETAILED RESULTS FOR {env_name.upper()}")
    print(f"{'=' * 60}")

    sorted_agents = sorted(results.items(),
                           key=lambda x: x[1]['final_performance'],
                           reverse=True)

    print(f"{'Agent':<20} {'Final Score':<12} {'Convergence':<12} {'Std Dev':<10}")
    print("-" * 60)

    for agent_type, data in sorted_agents:
        final_score = data['final_performance']
        convergence = data['convergence_episode']
        std_dev = np.std([np.mean(r[-50:]) for r in data['rewards']])

        print(f"{agent_type:<20} {final_score:<12.1f} {convergence:<12} {std_dev:<10.2f}")


def main():
    """
    Main function that runs all required tasks
    """
    print("REINFORCEMENT LEARNING TASK 1")
    print("=" * 60)
    print("1. Optimizing hyperparameters for MazeWater2")
    print("2. Solving MazeWater2 with optimized parameters")
    print("3. Creating and solving custom SpiralMaze environment")
    print("4. Generating focused convergence plots")
    print("=" * 60)

    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Step 1: Optimize hyperparameters for MazeWater2
    optimization_results = optimize_for_mazewater2(num_trials=30)

    # Step 2: Solve MazeWater2 with optimized parameters
    mazewater2_results = solve_mazewater2_with_best_params(optimization_results)

    # Step 3: Solve custom environment
    spiralmaze_results = solve_custom_environment()

    # Step 4: Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    # Plot MazeWater2 results
    print("Plotting MazeWater2 results...")
    plot_convergence_focused(mazewater2_results, "MazeWater2", focus_window=200)
    plot_full_training(mazewater2_results, "MazeWater2")

    # Plot SpiralMaze results
    print("Plotting SpiralMaze results...")
    plot_convergence_focused(spiralmaze_results, "SpiralMaze", focus_window=300)
    plot_full_training(spiralmaze_results, "SpiralMaze")

    # Step 5: Print detailed results
    print_detailed_results(mazewater2_results, "MazeWater2")
    print_detailed_results(spiralmaze_results, "SpiralMaze")

    # Step 6: Visualize learned policies
    print("\n" + "=" * 60)
    print("VISUALIZING LEARNED POLICIES")
    print("=" * 60)

    # Get best parameters for visualization
    best_agent_type = max(optimization_results.keys(),
                          key=lambda x: optimization_results[x]['best_score'])
    best_params = optimization_results[best_agent_type]['best_params']

    # Visualize MazeWater2 policy
    print("Visualizing MazeWater2 policy...")
    visualize_learned_policy(MazeWater2(), best_agent_type, best_params)

    # Visualize SpiralMaze policy
    print("Visualizing SpiralMaze policy...")
    visualize_learned_policy(SpiralMaze(), 'qlearning', {'alpha': 0.1, 'gamma': 0.95, 'epsilon': 0.1})

    print("\n" + "=" * 60)
    print("TASK 1 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Summary:")
    print(f"- MazeWater2 best agent: {best_agent_type}")
    print(f"- MazeWater2 best final score: {mazewater2_results[best_agent_type]['final_performance']:.1f}")
    print(
        f"- SpiralMaze best final score: {max(spiralmaze_results.values(), key=lambda x: x['final_performance'])['final_performance']:.1f}")
    print("- All plots have been generated")
    print("- Learned policies have been visualized")


if __name__ == "__main__":
    main()
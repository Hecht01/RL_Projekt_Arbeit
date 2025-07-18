import optuna
import numpy as np
from typing import Dict, Any, List
from Agents import RLAgentFactory


class OptunaOptimizer:
    """Optuna-based hyperparameter optimizer for RL agents"""

    def __init__(self, env, agent_type: str, num_runs: int = 3,
                 episodes: int = 500, optimization_metric: str = 'convergence_speed'):
        self.env = env
        self.agent_type = agent_type
        self.num_runs = num_runs
        self.episodes = episodes
        self.optimization_metric = optimization_metric
        self.best_params = None
        self.best_score = None

    def objective(self, trial):
        """Objective function for Optuna optimization"""
        # Define hyperparameter search space
        alpha = trial.suggest_float('alpha', 0.01, 0.5, log=True)
        gamma = trial.suggest_float('gamma', 0.8, 0.999)
        epsilon = trial.suggest_float('epsilon', 0.01, 0.3)
        epsilon_decay = trial.suggest_float('epsilon_decay', 0.99, 0.999)
        epsilon_min = trial.suggest_float('epsilon_min', 0.001, 0.05)

        # Train agent with these parameters
        scores = []
        convergence_episodes = []

        for run in range(self.num_runs):
            agent = RLAgentFactory.create_agent(
                self.agent_type,
                self.env,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min
            )

            rewards = agent.train(self.episodes)

            # Calculate metrics
            final_score = np.mean(rewards[-50:])  # Average of last 50 episodes
            convergence_episode = self._find_convergence_episode(rewards)

            scores.append(final_score)
            convergence_episodes.append(convergence_episode)

        avg_score = np.mean(scores)
        avg_convergence = np.mean(convergence_episodes)

        # Return metric based on optimization goal
        if self.optimization_metric == 'final_score':
            return avg_score
        elif self.optimization_metric == 'convergence_speed':
            # We want to minimize convergence time (negative because Optuna maximizes)
            return -avg_convergence
        elif self.optimization_metric == 'combined':
            # Combine final score and convergence speed
            normalized_score = avg_score / 100.0  # Normalize assuming max reward ~100
            normalized_convergence = 1.0 - (avg_convergence / self.episodes)
            return normalized_score + normalized_convergence
        else:
            return avg_score

    def _find_convergence_episode(self, rewards: List[float], threshold: float = 0.9) -> int:
        """Find the episode where the agent converges"""
        if len(rewards) < 50:
            return len(rewards) - 1

        # Calculate moving average to smooth out noise
        window_size = 25
        moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

        # Find the maximum reward achieved
        max_reward = np.max(moving_avg)
        target_reward = max_reward * threshold

        # Find first episode where moving average exceeds threshold
        for i, reward in enumerate(moving_avg):
            if reward >= target_reward:
                return i + window_size - 1

        return len(rewards) - 1

    def optimize(self, n_trials: int = 100, timeout: int = None,
                 direction: str = 'maximize') -> Dict[str, Any]:
        """Run Optuna optimization"""
        print(f"Starting optimization for {self.agent_type} with {n_trials} trials...")

        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        self.best_params = study.best_params
        self.best_score = study.best_value

        print(f"\nOptimization completed!")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': study
        }

    def get_best_params(self) -> Dict[str, Any]:
        """Get the best parameters found"""
        if self.best_params is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        return self.best_params

    def create_optimized_agent(self):
        """Create an agent with the best parameters"""
        if self.best_params is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")

        return RLAgentFactory.create_agent(
            self.agent_type,
            self.env,
            **self.best_params
        )


def optimize_multiple_agents(env, agent_types: List[str],
                             n_trials: int = 50, **optimizer_kwargs) -> Dict[str, Dict]:
    """Optimize multiple agent types"""
    results = {}

    for agent_type in agent_types:
        print(f"\n{'=' * 60}")
        print(f"Optimizing {agent_type.upper()}")
        print(f"{'=' * 60}")

        optimizer = OptunaOptimizer(env, agent_type, **optimizer_kwargs)
        optimization_result = optimizer.optimize(n_trials=n_trials)

        results[agent_type] = {
            'optimizer': optimizer,
            'best_params': optimization_result['best_params'],
            'best_score': optimization_result['best_score'],
            'study': optimization_result['study']
        }

    return results


def print_optimization_summary(optimization_results: Dict[str, Dict]):
    """Print summary of optimization results"""
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)

    sorted_results = sorted(
        optimization_results.items(),
        key=lambda x: x[1]['best_score'],
        reverse=True
    )

    print(f"{'Agent':<20} {'Best Score':<15} {'Best Parameters'}")
    print("-" * 80)

    for agent_type, result in sorted_results:
        best_score = result['best_score']
        best_params = result['best_params']

        # Format parameters for display
        params_str = ", ".join([f"{k}={v:.4f}" for k, v in best_params.items()])

        print(f"{agent_type:<20} {best_score:<15.4f} {params_str}")


def quick_optimize_for_mazewater2(env, agent_type: str = 'qlearning',
                                  n_trials: int = 30) -> Dict[str, Any]:
    """Quick optimization specifically for MazeWater2 environment"""
    print(f"Quick optimization for {agent_type} on MazeWater2...")

    optimizer = OptunaOptimizer(
        env=env,
        agent_type=agent_type,
        num_runs=5,
        episodes=800,
        optimization_metric='combined'
    )

    return optimizer.optimize(n_trials=n_trials)



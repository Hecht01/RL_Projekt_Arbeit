import optuna
import numpy as np
from typing import Dict, Any, List
from Agents import RLAgentFactory


class OptunaOptimizer:
    """Optuna-based hyperparameter optimizer for RL agents"""

    def __init__(self, env, agent_type: str, num_runs: int = 3, max_episodes: int = 1000):
        self.env = env
        self.agent_type = agent_type
        self.num_runs = num_runs
        self.max_episodes = max_episodes

    def objective(self, trial):
        """Objective: minimize episodes needed to reach 90% of max performance"""
        # Focused hyperparameter search
        alpha = trial.suggest_float('alpha', 0.05, 0.3)
        gamma = trial.suggest_float('gamma', 0.9, 0.99)
        epsilon = trial.suggest_float('epsilon', 0.1, 0.3)
        epsilon_decay = trial.suggest_float('epsilon_decay', 0.995, 0.999)

        convergence_episodes = []

        for _ in range(self.num_runs):
            agent = RLAgentFactory.create_agent(
                self.agent_type, self.env,
                alpha=alpha, gamma=gamma, epsilon=epsilon,
                epsilon_decay=epsilon_decay, epsilon_min=0.01
            )

            rewards = agent.train(self.max_episodes)
            convergence_episode = self._find_convergence_episode(rewards)
            convergence_episodes.append(convergence_episode)

        # Return negative average convergence episode (Optuna maximizes)
        return -np.mean(convergence_episodes)

    def _find_convergence_episode(self, rewards: List[float], threshold: float = 0.9) -> int:
        """Find episode where agent reaches 90% of peak performance"""
        if len(rewards) < 20:
            return len(rewards) - 1

        # Smooth with moving average
        window = 15
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        max_reward = np.max(smoothed)
        target = max_reward * threshold

        for i, reward in enumerate(smoothed):
            if reward >= target:
                return i + window - 1
        return len(rewards) - 1

    def optimize(self, n_trials: int = 50) -> Dict[str, Any]:
        """Run optimization to find fastest converging parameters"""
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        study.optimize(self.objective, n_trials=n_trials)

        return {
            'best_params': study.best_params,
            'convergence_episodes': -study.best_value
        }


def optimize_for_fastest_convergence(env, agent_type: str, n_trials: int = 50) -> Dict[str, Any]:
    """Quick function to optimize for fastest convergence"""
    optimizer = OptunaOptimizer(env, agent_type)
    return optimizer.optimize(n_trials)


def optimize_multiple_agents_minimal(env, agent_types: List[str], n_trials: int = 30) -> Dict[str, Dict]:
    """Optimize multiple agents for fastest convergence"""
    results = {}

    for agent_type in agent_types:
        print(f"Optimizing {agent_type} for fastest convergence...")
        result = optimize_for_fastest_convergence(env, agent_type, n_trials)
        results[agent_type] = result
        print(f"Best convergence: {result['convergence_episodes']:.1f} episodes")

    return results
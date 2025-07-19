import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional



class RLAgent:

    def __init__(self, env, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = np.zeros((env.num_states(), env.num_actions()))
        self.episode_rewards = []
        self.state_visits = np.zeros(env.num_states())
        self.state_action_visits = np.ones((env.num_states(), env.num_actions()))
        self.name = self.__class__.__name__

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int = None, done: bool = False):
        """Update the agent's knowledge"""
        pass

    def get_action(self, state: int, greedy: bool = False) -> int:
        """UCB action selection"""
        c = 2.0  # exploration constant

        # Ensure counts are safe
        total_visits = max(self.state_visits[state], 1)  # avoids log(0)
        action_visits = self.state_action_visits[state].copy()
        action_visits[action_visits < 1e-5] = 1e-5  # avoid division by 0

        # Compute UCB values
        ucb_values = self.Q[state] + c * np.sqrt(np.log(total_visits) / action_visits)
        return np.argmax(ucb_values)

    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, episodes: int = 1000, verbose: bool = False) -> List[float]:
        """Train the agent for specified number of episodes"""
        self.episode_rewards = []

        for episode in range(episodes):
            state = self.env.reset()

            cumulative_reward = 0
            action = self.get_action(state)  # For SARSA

            while True:
                next_state, reward, done = self.env.step(action)
                cumulative_reward += reward

                if done:
                    self.update(state, action, reward, next_state, done=True)
                    break
                self.state_visits[state] += 1
                self.state_action_visits[state, action] += 1
                next_action = self.get_action(next_state)
                self.update(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action

            self.episode_rewards.append(cumulative_reward)
            self.decay_epsilon()

            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, ε = {self.epsilon:.3f}")

        return self.episode_rewards

    def get_policy(self) -> np.ndarray:
        """Extract policy from Q-table"""
        return np.argmax(self.Q, axis=1)


class QLearningAgent(RLAgent):
    """Q-Learning implementation"""

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int = None, done: bool = False):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        self.Q[state, action] += self.alpha * (target - self.Q[state, action])


class SARSAAgent(RLAgent):
    """SARSA implementation"""

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int = None, done: bool = False):
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]

        self.Q[state, action] += self.alpha * (target - self.Q[state, action])


class ExpectedSARSAAgent(RLAgent):
    """Expected SARSA implementation"""

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int = None, done: bool = False):
        if done:
            target = reward
        else:
            # Expected value under current policy
            greedy_action = np.argmax(self.Q[next_state])
            expected_value = 0

            # Probability of taking greedy action
            prob_greedy = 1 - self.epsilon + self.epsilon / self.env.num_actions()
            expected_value += prob_greedy * self.Q[next_state, greedy_action]

            # Probability of taking random actions
            prob_random = self.epsilon / self.env.num_actions()
            for a in range(self.env.num_actions()):
                if a != greedy_action:
                    expected_value += prob_random * self.Q[next_state, a]

            target = reward + self.gamma * expected_value

        self.Q[state, action] += self.alpha * (target - self.Q[state, action])


class DoubleQLearningAgent(RLAgent):
    """Double Q-Learning implementation"""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.Q1 = np.zeros((env.num_states(), env.num_actions()))
        self.Q2 = np.zeros((env.num_states(), env.num_actions()))
        self.Q = self.Q1 + self.Q2  # For action selection

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int = None, done: bool = False):
        if done:
            target = reward
        else:
            if np.random.random() < 0.5:
                # Update Q1 using Q2
                best_action = np.argmax(self.Q1[next_state])
                target = reward + self.gamma * self.Q2[next_state, best_action]
                self.Q1[state, action] += self.alpha * (target - self.Q1[state, action])
            else:
                # Update Q2 using Q1
                best_action = np.argmax(self.Q2[next_state])
                target = reward + self.gamma * self.Q1[next_state, best_action]
                self.Q2[state, action] += self.alpha * (target - self.Q2[state, action])

        self.Q = self.Q1 + self.Q2  # Update combined Q for action selection


class ValueIterationAgent(RLAgent):
    """Value Iteration implementation (model-based)"""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.V = np.zeros(env.num_states())
        self.build_model()

    def build_model(self):
        n_states = self.env.num_states()
        n_actions = self.env.num_actions()

        self.transitions = np.zeros((n_states, n_actions, n_states))
        self.rewards = np.zeros((n_states, n_actions))

        for s in range(n_states):
            for a in range(n_actions):
                next_s, reward, done = self.env.step_dp(s, a)
                self.transitions[s, a, next_s] = 1.0
                self.rewards[s, a] = reward

    def value_iteration(self, theta: float = 1e-6, max_iterations: int = 1000):
        """Perform value iteration"""
        for i in range(max_iterations):
            delta = 0
            new_V = np.zeros_like(self.V)

            for s in range(self.env.num_states()):
                v = self.V[s]
                action_values = []

                for a in range(self.env.num_actions()):
                    action_value = self.rewards[s, a] + self.gamma * np.sum(
                        self.transitions[s, a] * self.V
                    )
                    action_values.append(action_value)

                new_V[s] = max(action_values)
                delta = max(delta, abs(v - new_V[s]))

            self.V = new_V
            if delta < theta:
                break

        # Extract Q-values and policy
        for s in range(self.env.num_states()):
            for a in range(self.env.num_actions()):
                self.Q[s, a] = self.rewards[s, a] + self.gamma * np.sum(
                    self.transitions[s, a] * self.V
                )

    def train(self, episodes: int = 1000, verbose: bool = False) -> List[float]:
        self.value_iteration()

        # Test the learned policy
        self.episode_rewards = []
        for _ in range(episodes):
            state = self.env.reset()
            cumulative_reward = 0

            while True:
                action = self.get_action(state, greedy=True)
                next_state, reward, done = self.env.step(action)
                cumulative_reward += reward

                if done:
                    break
                state = next_state

            self.episode_rewards.append(cumulative_reward)

        return self.episode_rewards

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int = None, done: bool = False):
        pass  # VI doesn't need runtime updates


class RLAgentFactory:
    """Factory class for creating RL agents"""

    _agents = {
        'qlearning': QLearningAgent,
        'sarsa': SARSAAgent,
        'expected_sarsa': ExpectedSARSAAgent,
        'double_qlearning': DoubleQLearningAgent,
        'value_iteration': ValueIterationAgent,
    }

    @classmethod
    def create_agent(cls, agent_type: str, env, **kwargs) -> RLAgent:
        """Create an RL agent of the specified type"""
        if agent_type.lower() not in cls._agents:
            raise ValueError(f"Unknown agent type: {agent_type}. "
                             f"Available types: {list(cls._agents.keys())}")

        return cls._agents[agent_type.lower()](env, **kwargs)

    @classmethod
    def get_available_agents(cls) -> List[str]:
        """Get list of available agent types"""
        return list(cls._agents.keys())


def compare_agents(env, agent_types: List[str], num_runs: int = 10,
                   episodes: int = 1000, **agent_kwargs) -> Dict:
    """Compare multiple RL agents on the same environment"""
    results = {}

    for agent_type in agent_types:
        print(f"\nTraining {agent_type}...")
        agent_results = []

        for run in range(num_runs):
            if num_runs > 1:
                print(f"  Run {run + 1}/{num_runs}")
            agent = RLAgentFactory.create_agent(agent_type, env, **agent_kwargs)
            rewards = agent.train(episodes)
            agent_results.append(rewards)

        results[agent_type] = {
            'rewards': np.array(agent_results),
            'mean_rewards': np.mean(agent_results, axis=0),
            'std_rewards': np.std(agent_results, axis=0),
            'final_performance': np.mean([r[-100:] for r in agent_results]),
            'convergence_episode': find_convergence_episode(agent_results),
        }

    return results


def find_convergence_episode(agent_results: List[List[float]], threshold: float = 0.9) -> int:
    """Find the episode where the agent converges to a good solution"""
    mean_rewards = np.mean(agent_results, axis=0)
    max_reward = np.max(mean_rewards)
    target_reward = max_reward * threshold

    for i, reward in enumerate(mean_rewards):
        if reward >= target_reward:
            return i

    return len(mean_rewards) - 1


def plot_comparison(results: Dict, title: str = "RL Agent Comparison",
                    focus_episodes: int = None, save_path: str = None):
    """Plot comparison of different RL agents"""
    plt.figure(figsize=(12, 8))

    for agent_type, data in results.items():
        mean_rewards = data['mean_rewards']
        std_rewards = data['std_rewards']
        episodes = range(len(mean_rewards))

        # Focus on specific episodes if requested
        if focus_episodes:
            episodes = episodes[:focus_episodes]
            mean_rewards = mean_rewards[:focus_episodes]
            std_rewards = std_rewards[:focus_episodes]

        plt.plot(episodes, mean_rewards, label=f"{agent_type} (final: {data['final_performance']:.1f})")
        plt.fill_between(episodes,
                         mean_rewards - std_rewards,
                         mean_rewards + std_rewards,
                         alpha=0.2)

    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_single_agent_convergence(results: Dict, agent_type: str,
                                  title: str = None, save_path: str = None):
    """Plot convergence of a single agent with focus on solving episodes"""
    if agent_type not in results:
        print(f"Agent type {agent_type} not found in results")
        return

    data = results[agent_type]
    mean_rewards = data['mean_rewards']
    std_rewards = data['std_rewards']
    convergence_episode = data['convergence_episode']

    # Focus on episodes around convergence
    start_episode = max(0, convergence_episode - 100)
    end_episode = min(len(mean_rewards), convergence_episode + 200)

    episodes = range(start_episode, end_episode)
    focused_mean = mean_rewards[start_episode:end_episode]
    focused_std = std_rewards[start_episode:end_episode]

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, focused_mean, label=f"{agent_type}", linewidth=2)
    plt.fill_between(episodes,
                     focused_mean - focused_std,
                     focused_mean + focused_std,
                     alpha=0.3)

    # Mark convergence point
    plt.axvline(x=convergence_episode, color='red', linestyle='--',
                label=f'Convergence (episode {convergence_episode})')

    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title(title or f"{agent_type} - Convergence Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def print_performance_summary(results: Dict):
    """Print performance summary for all agents"""
    print("PERFORMANCE SUMMARY")

    # Sort by final performance
    sorted_agents = sorted(results.items(),
                           key=lambda x: x[1]['final_performance'],
                           reverse=True)

    print(f"{'Agent':<20} {'Final Score':<12} {'Convergence':<12} {'Std Dev':<10}")
    print("-" * 60)

    for agent_type, data in sorted_agents:
        final_score = data['final_performance']
        convergence = data['convergence_episode']
        std_dev = np.std([np.mean(r[-100:]) for r in data['rewards']])

        print(f"{agent_type:<20} {final_score:<12.1f} {convergence:<12} {std_dev:<10.2f}")


def train_single_agent(env, agent_type: str, episodes: int = 1000,
                       num_runs: int = 10, **agent_kwargs) -> Dict:
    """Train a single agent type"""
    results = compare_agents(env, [agent_type], num_runs, episodes, **agent_kwargs)
    return results[agent_type]
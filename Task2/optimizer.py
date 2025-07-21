import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import cv2
import optuna
import os

from DQN import DQNAgent
from helper import CarRacingEnvironment

def train_agent(config, trial=None):
    """Train the DQN agent"""

    env = CarRacingEnvironment()
    agent = DQNAgent(
        state_size=(4, 84, 84),
        action_size=5,
        lr=config['lr'],
        gamma=config['gamma'],
        epsilon=config['epsilon'],
        epsilon_min=config['epsilon_min'],
        epsilon_decay=config['epsilon_decay'],
        memory_size=config['memory_size'],
        batch_size=config['batch_size'],
        target_update=config['target_update'],
        hidden_size=config['hidden_size']
    )

    episodes = config['episodes']
    scores = []
    recent_scores = deque(maxlen=5)
    best_avg_score = -np.inf

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        max_steps = 2000  # Prevent infinite episodes

        while step_count < max_steps:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step_count += 1

            if len(agent.memory) > agent.batch_size:
                agent.replay()

            if done:
                break

        scores.append(total_reward)
        recent_scores.append(total_reward)
        avg_recent_score = np.mean(recent_scores)

        if len(recent_scores) == 5 and avg_recent_score > best_avg_score:
            best_avg_score = avg_recent_score

        print(f"Episode {episode + 1}, Score: {total_reward:.1f}, "
              f"Avg Recent: {avg_recent_score:.1f}, Epsilon: {agent.epsilon:.3f}")

        # Early stopping condition
        if len(recent_scores) == 5 and avg_recent_score > 500:
            print(f"Solved in {episode + 1} episodes!")
            break

        # Pruning for Optuna
        if trial and episode > 100:
            trial.report(avg_recent_score, episode)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    env.close()
    return scores, agent, best_avg_score


def objective(trial):
    """Optuna objective function for hyperparameter optimization"""

    config = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'gamma': trial.suggest_float('gamma', 0.95, 0.999),
        'epsilon': trial.suggest_float('epsilon', 0.8, 1.0),
        'epsilon_min': trial.suggest_float('epsilon_min', 0.01, 0.1),
        'epsilon_decay': trial.suggest_float('epsilon_decay', 0.99, 0.999),
        'memory_size': trial.suggest_int('memory_size', 50000, 200000),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'target_update': trial.suggest_int('target_update', 500, 2000),
        'hidden_size': trial.suggest_categorical('hidden_size', [256, 512, 1024]),
        'episodes': 500
    }

    scores, agent, best_avg_score = train_agent(config, trial)
    return best_avg_score


def run_multiple_trainings(best_config, n_runs=10):
    """Run multiple training sessions with best config"""

    all_scores = []

    for run in range(n_runs):
        print(f"\n=== Training Run {run + 1}/{n_runs} ===")
        scores, agent, _ = train_agent(best_config)
        all_scores.append(scores)

        # Save best model from each run
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(agent.q_network.state_dict(), f'models/dqn_run_{run + 1}.pth')

    return all_scores


def plot_results(all_scores, save_path='training_results.png'):
    """Plot training results"""

    plt.figure(figsize=(12, 8))

    # Plot individual runs
    max_episodes = max(len(scores) for scores in all_scores)

    for i, scores in enumerate(all_scores):
        # Pad scores if needed
        padded_scores = scores + [scores[-1]] * (max_episodes - len(scores))
        plt.plot(range(1, len(padded_scores) + 1), padded_scores,
                 alpha=0.3, color='blue', linewidth=0.8)

    # Calculate and plot average
    avg_scores = []
    for episode in range(max_episodes):
        episode_scores = []
        for scores in all_scores:
            if episode < len(scores):
                episode_scores.append(scores[episode])
            else:
                episode_scores.append(scores[-1])
        avg_scores.append(np.mean(episode_scores))

    plt.plot(range(1, len(avg_scores) + 1), avg_scores,
             color='red', linewidth=2, label='Average over 10 runs')

    plt.axhline(y=500, color='green', linestyle='--',
                label='Target Score (500)', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('CarRacing-v2 Training Results (DQN)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
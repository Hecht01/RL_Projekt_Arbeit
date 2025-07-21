from optimizer import *
import pickle

def main():
    """Main training function"""

    print("Starting CarRacing DQN Training")
    print("================================")

    # Option 1: Use predefined config (faster)
    best_config = {
        'lr': 0.0001,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': 100000,
        'batch_size': 64,
        'target_update': 1000,
        'hidden_size': 512,
        'episodes': 1000
    }

    # Option 2: Hyperparameter optimization with Optuna (uncomment to use)
    """
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print("Best parameters:", study.best_params)
    best_config = study.best_params
    best_config['episodes'] = 1000
    """

    # Run multiple training sessions
    print("\nStarting multiple training runs...")
    all_scores = run_multiple_trainings(best_config, n_runs=10)

    # Plot results
    print("\nPlotting results...")
    plot_results(all_scores)

    # Save results
    with open('training_results.pkl', 'wb') as f:
        pickle.dump({
            'all_scores': all_scores,
            'best_config': best_config
        }, f)

    print("Training completed!")
    print("Models saved in 'models/' directory")
    print("Results saved as 'training_results.pkl'")
    print("Plot saved as 'training_results.png'")


if __name__ == "__main__":
    main()
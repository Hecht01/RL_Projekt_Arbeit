from optimizer import *
import pickle
import optuna

def main():
    """Main training function"""

    print("Starting CarRacing DQN Training")
    print("================================")


    best_config = {'lr': 0.00022310809611857726,
                    'gamma': 0.984220016687832,
                    'epsilon': 0.8069889245412875,
                    'epsilon_min': 0.03247386481896733,
                    'epsilon_decay': 0.9931884631252353,
                    'memory_size': 161594,
                    'batch_size': 32,
                    'target_update': 1594,
                    'hidden_size': 512
                }

    """
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print("Best parameters:", study.best_params)
    best_config = study.best_params
    best_config['episodes'] = 2500
    """

    best_config['episodes'] = 2500
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
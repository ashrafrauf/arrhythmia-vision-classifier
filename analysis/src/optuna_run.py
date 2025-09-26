# --- Deep learning packages ---
import optuna

# --- Utility packages ---
import datetime
import os
import json
import argparse
from functools import partial

# --- Helper functions ---
import ecg_utils



def objective_function(
        trial,
        experiment_dir,
        model_arch,
        dataset_config,
        freeze_backbone=True,
        num_workers=2,
        random_state=42,
        state_dict_path=None,
        stop_min_epoch=25,
        max_epoch=100,
        train_env='local'
):
    """
    Implements Bayesian optimisation search using Tree-structured Parzen Estimator (TPE)
    algorithm via Optuna. Saves results to a sqlite database to allow for interrupted runs.
    """
    # Identify trial number.
    print(f"\n[{datetime.datetime.now()}]    Running trial {trial.number}\n")
    # Identify training type.
    train_stage = "head" if freeze_backbone else "full"
    print(f"\nTraining stage: {train_stage}.")

    # Setup training paths.
    if train_env == 'local':
        if model_arch == 'efficientnetb3':
            lmdb_path = f'/Users/ashrafrauf/Documents/GitHub/arrhythmia-vision-classifier/data/processed-data/{dataset_config}/train/img_efficientnet.lmdb'
        else:
            lmdb_path = f'/Users/ashrafrauf/Documents/GitHub/arrhythmia-vision-classifier/data/processed-data/{dataset_config}/train/img_resnet.lmdb'
        csv_path = f'/Users/ashrafrauf/Documents/GitHub/arrhythmia-vision-classifier/data/processed-data/{dataset_config}/train/labels_keys.csv'
    
    elif train_env == 'slurm':
        if model_arch == 'efficientnetb3':
            lmdb_path = f'/users/userid/archive/ecg-cnn-data/processed-data/{dataset_config}/train/img_efficientnet.lmdb'
        else:
            lmdb_path = f'/users/userid/archive/ecg-cnn-data/processed-data/{dataset_config}/train/img_resnet.lmdb'
        csv_path = f'/users/userid/archive/ecg-cnn-data/processed-data/{dataset_config}/train/labels_keys.csv'

    # Setup Optuna paths.
    grid_results_dir = os.path.join(experiment_dir, "grid-results")
    model_cache_directory = os.path.join(experiment_dir, "final-models", "cache")
    model_best_directory = os.path.join(experiment_dir, "final-models", "best")
    os.makedirs(grid_results_dir, exist_ok=True)


    optimizer_name = trial.suggest_categorical("optimizer_name", ["AdamW", "SGD"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.0, 0.9) if optimizer_name == "SGD" else 0.9
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    scheduler_name = trial.suggest_categorical("scheduler_name", ["StepLR", "ReduceLROnPlateau", "OneCycleLR"])
    scheduler_step_size = trial.suggest_int("scheduler_step_size", 10, 30) if scheduler_name == "StepLR" else 20
    scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.1, 0.9) if scheduler_name == "StepLR" else 0.5
    scheduler_reduce_factor = trial.suggest_float("scheduler_reduce_factor", 0.1, 0.7) if scheduler_name == "ReduceLROnPlateau" else 0.1
    scheduler_reduce_patience = trial.suggest_int("scheduler_reduce_patience", 2, 10) if scheduler_name == "ReduceLROnPlateau" else 5

    model_label = f'trial_{trial.number}'

    # Run training loop.
    model, bestMetricsDict, metricsHist = ecg_utils.train_cv_model(
        lmdb_path,
        csv_path,
        model_arch,
        optimizer_name,
        model_cache_directory,
        model_best_directory,
        batchSize=batch_size,
        numWorkers=num_workers,
        randomState=random_state,
        trainIdx=None,
        valIdx=None,
        valSplitRatio=0.2,
        freezeBackbone=freeze_backbone,
        initMethod='None',
        stateDictPath=state_dict_path,
        learningRate=learning_rate,
        weightDecay=weight_decay,
        momentum=momentum,
        schedulerName=scheduler_name,
        schedulerStepSize=scheduler_step_size,
        schedulerGamma=scheduler_gamma,
        schedulerReduceFactor=scheduler_reduce_factor,
        schedulerReducePatience=scheduler_reduce_patience,
        evalMetric='f1',
        evalPatience=10,
        evalMinDelta=0.0,
        minEpochStop=stop_min_epoch,
        modelLabel=model_label,
        cacheInterval=20,
        verbose=False,
        numEpochs=max_epoch,
        saveCheckpoint=False,
        saveBestModel=False,
        earlyStop=True,
        cleanupInterval=20,
        optuna_trial=trial
    )

    # Add metadata to results.
    result_dict = {
        'stage': train_stage,
        'architecture': model_arch,
        'model': model_label,
        'loss': bestMetricsDict['loss'],
        'accuracy': bestMetricsDict['accuracy'],
        'f1': bestMetricsDict['f1'],
        'precision': bestMetricsDict['precision'],
        'recall': bestMetricsDict['recall'],
        'best_epoch': bestMetricsDict['epoch'],
        'optimizer': optimizer_name,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'scheduler': scheduler_name,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'scheduler_step_size': scheduler_step_size,
        'scheduler_gamma': scheduler_gamma,
        'scheduler_reduce_factor': scheduler_reduce_factor,
        'scheduler_reduce_patience': scheduler_reduce_patience,
        'earlystop_min_epoch': stop_min_epoch,
        'time_total_training': bestMetricsDict['timeTotalTraining'],
        'time_epoch_average': bestMetricsDict['timeAveragePerEpoch']
    }

    # Save results.
    output_filename = f'{model_label}_metrics.json'
    filepath = os.path.join(grid_results_dir, output_filename)
    with open(filepath, 'w') as f:
        json.dump(result_dict, f, indent=4)
    print(f'\n[{datetime.datetime.now()}]    Training evaluation metrics for best epoch is saved at {filepath}\n')

    return bestMetricsDict['f1']



#-------------- #
# Main Function #
#-------------- #
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_arch", type=str, default="resnet18", help="Model architecture used for the experiment.")
    parser.add_argument("--dataset_config", type=str, default='config1', help='Dataset config used for the experiment.')
    parser.add_argument("--num_trials", type=int, default=30, help='Number of search trials to run.')
    parser.add_argument("--freeze_backbone", action="store_true", help="Freezes all layers except classification head.")
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false", help="Do not freeze any layers.")
    parser.add_argument("--random_state", type=int, default=42, help='Number to initialise random seed.')
    parser.add_argument("--state_dict_folder", type=str, default="None", help="Path to state dict to load previously trained weights.")
    parser.add_argument("--train_env", type=str, default="local", help='Indicator whether training on HPC (i.e., slurm) or Macbook (i.e., local)')
    parser.add_argument("--num_workers", type=int, default=0, help='Number of PyTorch DataLoader workers.')
    parser.set_defaults(freeze_backbone=True)
    args = parser.parse_args()

    # Set global seeds.
    ecg_utils.set_global_seeds(seed_num=args.random_state)

    # Identify training type.
    train_stage = "head" if args.freeze_backbone else "full"

    # Identify scipt location and project directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Optuna setup.
    study_name = f'optuna_ecg_{args.model_arch}_{train_stage}_{args.dataset_config}'

    # Setup Optuna directory.
    main_optuna_dir = os.path.join(project_dir, "optuna")
    experiment_dir = os.path.join(main_optuna_dir, f'{args.model_arch}_{train_stage}_{args.dataset_config}')
    optuna_res_dir = os.path.join(experiment_dir, "optuna-results")
    os.makedirs(optuna_res_dir, exist_ok=True)

    # Use SQLite file as persistent storage
    storage_path = f"sqlite:///{os.path.join(optuna_res_dir, 'optuna_study.db')}"
    storage = optuna.storages.RDBStorage(
        url=storage_path,
        engine_kwargs={"connect_args": {"timeout": 300}}  # Prevent locking issues
    )

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=args.random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10, interval_steps=1, n_min_trials=3),
        direction="maximize",
        load_if_exists=True
    )

    # Run Bayesian search.
    if args.state_dict_folder == "None":
        state_dict_path = None
    else:
        state_dict_path = os.path.join(project_dir, "experiments", args.state_dict_folder, "final-models", "best", "best_model.pth")

    study.optimize(
        partial(
            objective_function,
            experiment_dir = experiment_dir,
            model_arch = args.model_arch,
            dataset_config = args.dataset_config,
            freeze_backbone = args.freeze_backbone,
            num_workers=args.num_workers,
            random_state = args.random_state,
            state_dict_path=state_dict_path,
            stop_min_epoch=20,
            max_epoch=50,
            train_env=args.train_env
        ),
        n_trials=args.num_trials,
        timeout=None,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=False
    )

    # Save best trial.
    best_trial = study.best_trial
    best_trial_path = os.path.join(optuna_res_dir, "best_trial.json")
    with open(best_trial_path, "w") as f:
        json.dump({
            "trial_number": best_trial.number,
            "f1": best_trial.value,
            "params": best_trial.params
        }, f, indent=4)
    print(f"\n🏆 Best trial saved to: {best_trial_path}")

    # Save final study summary (optional)
    study_summary_path = os.path.join(optuna_res_dir, "study_summary.txt")
    with open(study_summary_path, "w") as f:
        f.write(f"Study: {study_name}\n")
        f.write(f"Trials: {len(study.trials)}\n")
        f.write(f"Best Trial: #{best_trial.number}\n")
        f.write(f"F1 Score: {best_trial.value:.4f}\n")
        f.write(f"Params: {json.dumps(best_trial.params, indent=4)}\n")
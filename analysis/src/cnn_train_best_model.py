# --- Utility packages ---
import os
import argparse
import json
import subprocess
import datetime

# --- Helper functions ---
import ecg_utils



def get_best_config_and_train_cnn(
        experiment_folder,
        path_to_lmdb,
        path_to_csv,
        modelLabel='best_model',
        bestMetric='f1',
        bestAggMethod='mean',
        hyperparamsList=['optimizer', 'learning_rate', 'weight_decay', 'scheduler', 'batch_size', 'init_method']
):
    """
    Consolidates results for a specific grid search space, identifies the best hyperparameter combination,
    and re-runs the training procedure using the best hyperparameter combination. Saves the outputs
    within the same experiment folder.
    """
    # Set global seeds.
    ecg_utils.set_global_seeds(seed_num=42)

    # Identify script location and project directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Identify relevant folders.
    path_to_grid_results = os.path.join(project_dir, "experiments", experiment_folder, "grid-results")
    path_to_configs = os.path.join(project_dir, "experiments", experiment_folder, "configs")
    path_to_logs = os.path.join(project_dir, "experiments", experiment_folder, "local-logs")
    path_to_train_script = os.path.join(project_dir, "src", "cnn_train.py")

    # Consolidate results.
    results_consol_df, results_smry_df = ecg_utils.combine_fold_results(path_to_grid_results)

    # Get best config label.
    best_config_row = ecg_utils.get_best_config(
        results_smry_df,
        bestMetric=bestMetric,
        bestAggMethod=bestAggMethod,
        hyperparamsList=hyperparamsList
    )
    best_config_label = best_config_row.model_num

    # Get best configs.
    grid_config_filepath = os.path.join(path_to_configs, "slurm_job_array_manifest.json")
    with open(grid_config_filepath, 'r') as file:
        grid_config_dict = json.load(file)
    
    best_config = [config for config in grid_config_dict if config['config_label']==best_config_label][0]  # Only select the first value, the rest are similar values for diffferent folds.
    print(f'Best config selection:\n{best_config}')

    # Save best configs to disk.
    best_config_filepath = os.path.join(path_to_configs, "best_model_config.json")
    with open(best_config_filepath, 'w') as f:
        json.dump(best_config, f, indent=4)
    print(f'\nBest config ({best_config_label}) saved here: {best_config_filepath}')

    # Define training arguments.
    train_command = [
        "python", path_to_train_script,
        # Preliminary arguments.
        "--config_label", modelLabel,
        "--val_ratio", str(0.2),
        "--lmdb_path", path_to_lmdb,
        "--csv_path", path_to_csv,
        "--dataset_ver", best_config['dataset_ver'],
        "--submit_timestamp", best_config['submit_timestamp'],
        # Architecture arguments.
        "--model_arch", best_config['model_arch'],
        "--layer_init", best_config['layer_init'],
        "--optimizer_name", best_config['optimizer_name'],
        "--opt_learning_rate", str(best_config['opt_learning_rate']),
        "--opt_weight_decay", str(best_config['opt_weight_decay']),
        "--scheduler_name", best_config['scheduler_name'],
        "--scheduler_step", str(best_config.get('scheduler_step', 20)),
        "--scheduler_gamma", str(best_config.get('scheduler_gamma',0.5)),
        "--scheduler_reduce_factor", str(best_config.get('scheduler_reduce_factor', 0.1)),
        "--scheduler_reduce_patience", str(best_config.get('scheduler_reduce_patience', 5)),
        # Training arguments.
        "--batch_size", str(best_config['batch_size']),
        "--max_epoch", str(best_config['max_epoch']),
        "--eval_metric", str(best_config['eval_metric']),
        "--stop_patience", str(best_config['stop_patience']),
        "--stop_min_delta", str(best_config['stop_min_delta']),
        "--stop_min_epoch", str(best_config['stop_min_epoch']),
        "--num_workers", str(best_config['num_workers']),
        "--random_state", str(best_config['random_state']),
        # Utility arguments.
        "--save_checkpoints",
        "--save_bestmodel",
        "--checkpoint_interval", str(best_config['checkpoint_interval']),
        "--cleanup_interval", str(best_config['cleanup_interval']),
        "--verbose_flag"
    ]

    # Handle boolean arguments.
    train_command.extend(["--freeze_backbone" if best_config['freeze_backbone'] else "--no_freeze_backbone"])
    train_command.extend(["--early_stop" if best_config['early_stop'] else "--no_early_stop"])
    
    # Add state dict folder if provided, for fine-tuning training.
    if best_config.get('state_dict_folder', None) is not None:
        train_command.extend(["--state_dict_folder", best_config['state_dict_folder']])

    # Define paths for log outputs.
    current_time = datetime.datetime.now()
    time_suffix = current_time.strftime("%y%m%d_%H%M")
    log_out_path = os.path.join(path_to_logs, "out", f'{time_suffix}_train_best_model.out')
    log_err_path = os.path.join(path_to_logs, "err", f'{time_suffix}_train_best_model.err')

    # Run training loop.
    with open(log_out_path, "w") as out_file, open(log_err_path, "w") as err_file:
        train_process = subprocess.Popen(
            train_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Decodes stdout/stderr as text.
            bufsize=1  # Buffers one line at a time.
        )

        # Read stdout line by line.
        for line in train_process.stdout:
            print(line, end="")
            out_file.write(line)
        
        # Read stderr line by line.
        for line in train_process.stderr:
            print(line, end="")
            err_file.write(line)
        
        train_process.wait()

        if train_process.returncode != 0:
            raise subprocess.CalledProcessError(train_process.returncode, train_command)





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_folder", type=str, required=True, help='Path to the specific experiment folder created when generating hyperparameter configs.')
    parser.add_argument("--model_label", type=str, default="best_model", help='Filename for model state dict object.')
    parser.add_argument("--best_metric", type=str, default="f1", help='Evaluation metric to be used to select the best hyperparameter configuration.')
    parser.add_argument("--best_agg_method", type=str, default="mean", help='Summary statistic used to select the best hyperparameter configuration.')
    parser.add_argument("--train_env", type=str, default="local", help='Runtime environment')
    args = parser.parse_args()

    experiment_folder = args.experiment_folder
    experiment_metadata = experiment_folder.split("_")
    dataset_config = experiment_metadata[-1]
    experiment_model = experiment_metadata[-3]

    if args.train_env=='local':
        if experiment_model == 'efficientnetb3':
            lmdb_path = f'/Users/ashrafrauf/Documents/GitHub/arrhythmia-vision-classifier/data/processed-data/{dataset_config}/train/img_efficientnet.lmdb'
        else:
            lmdb_path = f'/Users/ashrafrauf/Documents/GitHub/arrhythmia-vision-classifier/data/processed-data/{dataset_config}/train/img_resnet.lmdb'
        
        csv_path = f'/Users/ashrafrauf/Documents/GitHub/arrhythmia-vision-classifier/data/processed-data/{dataset_config}/train/labels_keys.csv'
    
    elif args.train_env=='slurm':
        if experiment_model == 'efficientnetb3':
            lmdb_path = f'/users/userid/archive/ecg-cnn-data/processed-data/{dataset_config}/train/img_efficientnet.lmdb'
        else:
            lmdb_path = f'/users/userid/archive/ecg-cnn-data/processed-data/{dataset_config}/train/img_resnet.lmdb'
        
        csv_path = f'/users/userid/archive/ecg-cnn-data/processed-data/{dataset_config}/train/labels_keys.csv'

    print(f'Training best model for experiment: {args.experiment_folder}\n')
    print(f'Training done on: {args.train_env}\n')

    get_best_config_and_train_cnn(
        args.experiment_folder,
        lmdb_path,
        csv_path,
        modelLabel=args.model_label,
        bestMetric=args.best_metric,
        bestAggMethod=args.best_agg_method
    )
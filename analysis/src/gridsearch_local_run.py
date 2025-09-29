# --- Utility packages ---
import os
import json
import datetime
import subprocess
import argparse

# --- Helper functions ---
import ecg_utils



def grid_search_local(
        experiment_folder,
        experiment_model,
        dataset_config
):
    """
    Implements grid search optimisation on the local machine. Due to unexpected behaviour of 
    torch_shm_manager on Macbook with MPS backend, the script is run within Python's subprocess
    function. Otherwise, torch_shm_manager will persist and cause memory bloat.
    """
    # Set global seeds.
    ecg_utils.set_global_seeds(seed_num=42)

    # Identify scipt location and project directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    main_data_dir = os.path.join(os.path.dirname(project_dir), 'data')

    # Identify relevant folders.
    path_to_grid_results = os.path.join(project_dir, "experiments", experiment_folder, "grid-results")
    path_to_configs = os.path.join(project_dir, "experiments", experiment_folder, "configs")
    path_to_logs = os.path.join(project_dir, "experiments", experiment_folder, "local-logs")
    path_to_train_script = os.path.join(project_dir, "src", "cnn_train.py")

    # Get dataset paths.
    path_to_lmdb, path_to_csv = ecg_utils.get_dataset_paths(main_data_dir, experiment_model, dataset_config, 'train')

    # Get hyperparameter grid.
    grid_config_filepath = os.path.join(path_to_configs, "slurm_job_array_manifest.json")
    with open(grid_config_filepath, 'r') as file:
        grid_config_dict = json.load(file)
    
    for config in grid_config_dict:
        print(f'\nTraining for {config['config_label']}  |  Fold {config['fold']} out of {config['total_folds']}  |  Timestamp: {datetime.datetime.now()}\n')

        # Define training arguments.
        train_command = [
            "python", path_to_train_script,
            # Preliminary arguments.
            "--config_label", f"{config['config_label']}",
            "--fold", str(config['fold']),
            "--total_folds", str(config['total_folds']),
            "--lmdb_path", path_to_lmdb,
            "--csv_path", path_to_csv,
            "--dataset_ver", config['dataset_ver'],
            "--submit_timestamp", config['submit_timestamp'],
            # Architecture arguments.
            "--model_arch", config['model_arch'],
            "--layer_init", config['layer_init'],
            "--optimizer_name", config['optimizer_name'],
            "--opt_learning_rate", str(config['opt_learning_rate']),
            "--opt_weight_decay", str(config['opt_weight_decay']),
            "--scheduler_name", config['scheduler_name'],
            "--scheduler_step", str(config['scheduler_step']),
            "--scheduler_gamma", str(config['scheduler_gamma']),
            # Training arguments.
            "--batch_size", str(config['batch_size']),
            "--max_epoch", str(config['max_epoch']),
            "--eval_metric", str(config['eval_metric']),
            "--stop_patience", str(config['stop_patience']),
            "--stop_min_delta", str(config['stop_min_delta']),
            "--stop_min_epoch", str(config['stop_min_epoch']),
            "--num_workers", str(config['num_workers']),
            "--random_state", str(config['random_state']),
            # Utility arguments.
            "--no_save_bestmodel",
            "--checkpoint_interval", str(config['checkpoint_interval']),
            "--cleanup_interval", str(config['cleanup_interval'])
        ]

        # Handle boolean arguments.
        train_command.extend(["--verbose_flag" if config['verbose_flag'] else "--no_verbose_flag"])
        train_command.extend(["--freeze_backbone" if config['freeze_backbone'] else "--no_freeze_backbone"])
        train_command.extend(["--early_stop" if config['early_stop'] else "--no_early_stop"])
        train_command.extend(["--save_checkpoints" if config['save_checkpoints'] else "--no_save_checkpoints"])
        train_command.extend(["--save_bestmodel" if config['save_bestmodel'] else "--no_save_bestmodel"])
        
        if config['state_dict_folder'] is not None:
            train_command.extend(["--state_dict_folder", config['state_dict_folder']])

        # Define paths for log outputs.
        current_time = datetime.datetime.now()
        time_suffix = current_time.strftime("%y%m%d_%H%M")
        log_out_path = os.path.join(path_to_logs, "out", f'{time_suffix}_gridsearch_{config['config_label']}_fold{config['fold']}.out')
        log_err_path = os.path.join(path_to_logs, "err", f'{time_suffix}_gridsearch_{config['config_label']}_fold{config['fold']}.err')

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
    args = parser.parse_args()

    experiment_folder = args.experiment_folder
    experiment_metadata = experiment_folder.split("_")
    dataset_config = experiment_metadata[-1]
    experiment_model = experiment_metadata[-3]

    grid_search_local(
        args.experiment_folder,
        experiment_model,
        dataset_config,
    )
# --- Utility packages --
import yaml
from itertools import product
import datetime
import os
import json

# --- Helper functions ---
import ecg_utils



def generate_grid_search_configs(
        grid_config_filepath,
        seed_num=42
):
    """
    Generates a list of dictionaries, with each dictionary containing a specific combination of hyperparameters.
    The list of dictionaries could then be used to implement grid search on SLURM by running parallel jobs, each
    job for a specific combination of hyperparameter for one fold.

    Arguments:
        grid_config_filepath (str): Path to the config template.
        seed_num (int): Number to initialise random seed. (Default: 42)
    """
    # Set global seeds.
    ecg_utils.set_global_seeds(seed_num=seed_num)

    # Load configs.
    with open(grid_config_filepath, "r") as f:
        grid_configs_dict = yaml.safe_load(f)
    
    # Load hyperparameter values.
    optimizer_list = grid_configs_dict['optimizer_name']
    learning_rate_list = grid_configs_dict['opt_learning_rate']
    weight_decay_list = grid_configs_dict['opt_weight_decay']
    scheduler_list = grid_configs_dict['scheduler_name']
    batch_size_list = grid_configs_dict['batch_size']
    layer_init_list = grid_configs_dict['layer_init']

    # Define number of folds.
    total_folds = grid_configs_dict.get('total_folds', 1)

    # Cartesian product all hyperparameter values.
    hyperparam_grid = list(product(optimizer_list, learning_rate_list, weight_decay_list, scheduler_list, batch_size_list, layer_init_list))

    # Extract current time to differentiate experiment.
    current_time = datetime.datetime.now()
    time_suffix = current_time.strftime("%y%m%d_%H%M")

    fixed_params = {
        # Preliminary arguments.
        'submit_timestamp': time_suffix,
        'dataset_ver': grid_configs_dict['dataset_ver'],
        'lmdb_path': grid_configs_dict['lmdb_path'],
        'csv_path': grid_configs_dict['csv_path'],
        'val_ratio': grid_configs_dict.get('val_ratio', 0.2),  # For non-kfold runs. Otherwise, ignored.
        # Architecture arguments.
        'model_arch': grid_configs_dict['model_arch'],
        'freeze_backbone': grid_configs_dict['freeze_backbone'],
        'state_dict_folder': grid_configs_dict.get('state_dict_folder', None),
        'scheduler_step': grid_configs_dict.get('scheduler_step', 20),
        'scheduler_gamma': grid_configs_dict.get('scheduler_gamma', 0.5),
        'scheduler_reduce_factor': grid_configs_dict.get('scheduler_reduce_factor', 0.1),
        'scheduler_reduce_patience':grid_configs_dict.get('scheduler_reduce_patience', 5),
        # Training arguments.
        'max_epoch': grid_configs_dict.get('max_epoch', 200),
        'eval_metric': grid_configs_dict.get('eval_metric', 'f1'),
        'early_stop': grid_configs_dict.get('early_stop', True),
        'stop_patience': grid_configs_dict.get('stop_patience', 10),
        'stop_min_delta': grid_configs_dict.get('stop_min_delta', 0.0),
        'stop_min_epoch': grid_configs_dict.get('stop_min_epoch', 25),
        'num_workers': grid_configs_dict.get('num_workers', 6),
        'random_state': grid_configs_dict.get('random_state', 42),
        # Utility arguments.
        'save_checkpoints': grid_configs_dict.get('save_checkpoints', False),
        'save_bestmodel': grid_configs_dict.get('save_checkpoints', False),
        'checkpoint_interval': grid_configs_dict.get('checkpoint_interval', 20),  # Ignored if save_checkpoints is False.
        'cleanup_interval': grid_configs_dict.get('cleanup_interval', 0),
        'verbose_flag': grid_configs_dict.get('verbose_flag', False)
    }

    # Generate all combinations of hyperparameters and folds.
    all_configs = []
    for i, (optimizer_name, learn_rate, weight_decay, scheduler_name, batch_size, layer_init) in enumerate(hyperparam_grid):
        for fold_count in range(total_folds):
            fold_config = {
                'config_label': f'config{i:04d}',
                'fold': fold_count,
                'total_folds': total_folds,
                'optimizer_name': optimizer_name,
                'opt_learning_rate': learn_rate,
                'opt_weight_decay': weight_decay,
                'scheduler_name': scheduler_name,
                'batch_size': batch_size,
                'layer_init': layer_init
            }
            # Add fixed parameters to each config.
            fold_config.update(fixed_params)
            all_configs.append(fold_config)
    
    print('\n---------------------------------')
    print('---------------------------------\n')
    print(f"Total config combinations: {len(hyperparam_grid)}")
    print(f"Number of folds per combination: {total_folds}")
    print(f"Total experiments: {len(all_configs)}")
    print('\n---------------------------------')
    print('---------------------------------\n')

    # Identify project root path.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Identify training type.
    train_stage = "head" if grid_configs_dict['freeze_backbone'] else "full"

    # Create directories to store experiment output.
    experiment_folder_name = f'{time_suffix}_{grid_configs_dict['model_arch']}_{train_stage}_{grid_configs_dict['dataset_ver']}'
    experiment_dir = os.path.join(project_dir, "experiments", experiment_folder_name)
    experiment_config_dir = os.path.join(experiment_dir, "configs")
    experiment_cache_dir = os.path.join(experiment_dir, "final-models", "cache")
    experiment_bestmodel_dir = os.path.join(experiment_dir, "final-models", "best")
    experiment_finalresults_dir = os.path.join(experiment_dir, "final-results")
    experiment_gridresults_dir = os.path.join(experiment_dir, "grid-results")
    experiment_slurmlogs_errdir = os.path.join(experiment_dir, "slurm-logs", "err")
    experiment_slurmlogs_outdir = os.path.join(experiment_dir, "slurm-logs", "out")
    experiment_locallogs_errdir = os.path.join(experiment_dir, "local-logs", "err")
    experiment_locallogs_outdir = os.path.join(experiment_dir, "local-logs", "out")
    
    os.makedirs(experiment_config_dir, exist_ok=True)
    os.makedirs(experiment_cache_dir, exist_ok=True)
    os.makedirs(experiment_bestmodel_dir, exist_ok=True)
    os.makedirs(experiment_finalresults_dir, exist_ok=True)
    os.makedirs(experiment_gridresults_dir, exist_ok=True)
    os.makedirs(experiment_slurmlogs_errdir, exist_ok=True)
    os.makedirs(experiment_slurmlogs_outdir, exist_ok=True)
    os.makedirs(experiment_locallogs_errdir, exist_ok=True)
    os.makedirs(experiment_locallogs_outdir, exist_ok=True)

    # Store config file.
    main_grid_filepath = os.path.join(experiment_config_dir, "experiment_hyperparameter_grid.json")
    with open(main_grid_filepath, 'w') as f:
        json.dump(grid_configs_dict, f, indent=4)
    print(f"Hyperparameter grid copied to: {main_grid_filepath}\n")

    # Store job array manifest.
    manifest_filepath = os.path.join(experiment_config_dir, "slurm_job_array_manifest.json")
    with open(manifest_filepath, 'w') as f:
        json.dump(all_configs, f, indent=4)
    
    print(f"Job array manifest saved to: {manifest_filepath}\n")





#-------------- #
# Main Function #
#-------------- #
if __name__ == "__main__":
    
    # Identify project root path.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Identify config filepath.
    config_folder = os.path.join(project_dir, "config-template")
    config_filename = "hyperparameter_grid_template.yaml"
    
    grid_config_filepath = os.path.join(config_folder, config_filename)

    generate_grid_search_configs(
        grid_config_filepath,
        seed_num=42
    )
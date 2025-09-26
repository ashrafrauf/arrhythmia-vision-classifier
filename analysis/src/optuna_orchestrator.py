# --- Utility packages ---
import os
import sys
import datetime
import subprocess
import argparse

# --- Helper functions ---
import ecg_utils

def run_optuna_in_subprocess(
        model_arch,
        dataset_config,
        num_trials,
        freeze_backbone=True,
        random_state=42,
        state_dict_folder="None",
        train_env="local",
        num_workers=0
):
    """
    An orchestrator to implement Bayesian optimisation search. This is due to unexpected 
    behaviour of torch_shm_manager on Macbook with MPS backend, the script is run within
    Python's subprocess function. Otherwise, torch_shm_manager will persist and cause
    memory bloat.

    """
    # Set global seeds.
    ecg_utils.set_global_seeds(seed_num=random_state)

    # Identify training type.
    train_stage = "head" if freeze_backbone else "full"

    # Identify scipt location and project directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    main_optuna_dir = os.path.join(project_dir, "optuna")
    experiment_dir = os.path.join(main_optuna_dir, f'{model_arch}_{train_stage}_{dataset_config}')

    # Identify relevant folders.
    path_to_train_script = os.path.join(project_dir, "src", "optuna_run.py")
    path_to_logs_dir = os.path.join(experiment_dir, "local-logs")
    os.makedirs(os.path.join(path_to_logs_dir, "out"), exist_ok=True)
    os.makedirs(os.path.join(path_to_logs_dir, "err"), exist_ok=True)
    print(f"\nLog files saved at: {path_to_logs_dir}\n")

    # Define paths for log outputs.
    current_time = datetime.datetime.now()
    time_suffix = current_time.strftime("%y%m%d_%H%M")
    log_out_path = os.path.join(path_to_logs_dir, "out", f'{time_suffix}.out')
    log_err_path = os.path.join(path_to_logs_dir, "err", f'{time_suffix}.err')

    # Define training arguments.
    train_command = [
        sys.executable, path_to_train_script,
        "--model_arch", model_arch,
        "--dataset_config", dataset_config,
        "--num_trials", str(num_trials),
        "--random_state", str(random_state),
        "--state_dict_folder", state_dict_folder,
        "--train_env", train_env,
        "--num_workers", str(num_workers)
    ]

    # Handle boolean arguments.
    train_command.extend(["--freeze_backbone" if freeze_backbone else "--no_freeze_backbone"])

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

    run_optuna_in_subprocess(
        args.model_arch,
        args.dataset_config,
        args.num_trials,
        args.freeze_backbone,
        args.random_state,
        args.state_dict_folder,
        args.train_env,
        num_workers=args.num_workers
    )
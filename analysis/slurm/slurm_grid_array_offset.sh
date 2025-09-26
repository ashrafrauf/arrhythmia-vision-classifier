#!/bin/bash
#SBATCH --job-name=mobilenetv3small_full_config1_grid                          # Job name for the array
#SBATCH --nodes=1                                           # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                                 # Run one task
#SBATCH --cpus-per-task=6                                   # CPUs per task (adjust based on cnn_train.py's num_workers)
#SBATCH --gres=gpu:1                                        # Request 1 GPU per task (adjust or remove if not using GPUs)
#SBATCH --mem=32G                                           # Memory per task (adjust)
#SBATCH --time=0-07:00:00                                   # Max time for each task hrs:min:sec
#SBATCH --partition=gengpu                                  # Your SLURM partition
#SBATCH --constraint=cuda12
#SBATCH --mail-type=ALL                                     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=username@citystgeorges.ac.uk    # Where to send mail
#SBATCH --output=/users/userid/ecg-cnn-home/experiments/250830_0214_mobilenetv3small_full_config1/slurm-logs/out/slurm-%A_%a.out   # <--- UPDATE EXPERIMENT FOLDER !!!
#SBATCH --error=/users/userid/ecg-cnn-home/experiments/250830_0214_mobilenetv3small_full_config1/slurm-logs/err/slurm-%A_%a.err    # <--- UPDATE EXPERIMENT FOLDER !!!
#SBATCH --array=0-159                                                                                                     # <--- UPDATE NUMBER OF GRID CONFIGS * FOLDS !!!


# -- Context ---
# Orchestrates the grid search optimisation within City University's Hyperion.
# Submits as a batch array job, where each element within the array is considered as a single job.
# This allows parallelisation of the grid search runs.
# Offset variable allows job to restart at selected points within the array, in case of interrupted runs.
# Script was built based on examples here: https://cityuni.service-now.com/sp?id=kb_article_view&sysparm_article=KB0012621


# --- Define Base Paths ---
PROJECT_ROOT="/users/userid/ecg-cnn-home"
EXPERIMENT_FOLDER="250830_0214_mobilenetv3small_full_config1"                                                                       # <--- UPDATE EXPERIMENT FOLDER !!!

# --- Define SLURM Array Config File ---
MANIFEST_FILE="$PROJECT_ROOT/experiments/$EXPERIMENT_FOLDER/configs/slurm_job_array_manifest.json"

# --- Check SLURM_ARRAY_TASK_ID against manifest length ---
NUM_CONFIGS=$(jq length "$MANIFEST_FILE")
if (( SLURM_ARRAY_TASK_ID >= NUM_CONFIGS )); then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of bounds."
    echo "Manifest file '$MANIFEST_FILE' only contains ${NUM_CONFIGS} configurations (indices 0 to $((NUM_CONFIGS - 1)))."
    exit 1
fi

# --- Setup Environment ---
# Enable modules.
source /opt/flight/etc/setup.sh
flight env activate gridware

#Remove any unwanted modules
module purge

# Initialize Conda.
source /users/userid/miniconda3/etc/profile.d/conda.sh
conda activate /users/userid/archive/envs/inm363_project

# Load modules required.
module load libs/nvidia-cuda/12.2.2/bin


# --- Navigate to Project Root ---
# This ensures cnn_train.py and ecg_utils are found correctly
cd $PROJECT_ROOT

# --- Get Task Configuration ---
# Get offset-adjusted index (relevant for indices > 1000).
ACTUAL_INDEX=$((SLURM_ARRAY_TASK_ID + OFFSET))

# Read the manifest file and extract the specific config for this array task. This uses 'jq' for JSON parsing.
TASK_CONFIG=$(jq ".[$ACTUAL_INDEX]" $MANIFEST_FILE)

# Extract arguments from the JSON config for the current task.
## Preliminaries args.
CONFIG_LABEL=$(echo $TASK_CONFIG | jq -r '.config_label')
FOLD=$(echo $TASK_CONFIG | jq -r '.fold')
TOTAL_FOLDS=$(echo $TASK_CONFIG | jq -r '.total_folds')
VAL_RATIO=$(echo $TASK_CONFIG | jq -r '.val_ratio')
LMDB_PATH=$(echo $TASK_CONFIG | jq -r '.lmdb_path')
CSV_PATH=$(echo $TASK_CONFIG | jq -r '.csv_path')
DATASET_VER=$(echo $TASK_CONFIG | jq -r '.dataset_ver')
SUBMIT_TIMESTAMP=$(echo $TASK_CONFIG | jq -r '.submit_timestamp')
## Architecture args.
MODEL_ARCH=$(echo $TASK_CONFIG | jq -r '.model_arch')
FREEZE_BACKBONE=$(echo $TASK_CONFIG | jq -r '.freeze_backbone')
STATE_DICT_FOLDER=$(echo $TASK_CONFIG | jq -r '.state_dict_folder')
LAYER_INIT=$(echo $TASK_CONFIG | jq -r '.layer_init')
OPTIMIZER_NAME=$(echo $TASK_CONFIG | jq -r '.optimizer_name')
OPT_LEARNING_RATE=$(echo $TASK_CONFIG | jq -r '.opt_learning_rate')
OPT_WEIGHT_DECAY=$(echo $TASK_CONFIG | jq -r '.opt_weight_decay')
SCHEDULER_NAME=$(echo $TASK_CONFIG | jq -r '.scheduler_name')
SCHEDULER_STEP=$(echo $TASK_CONFIG | jq -r '.scheduler_step')
SCHEDULER_GAMMA=$(echo $TASK_CONFIG | jq -r '.scheduler_gamma')
SCHEDULER_REDUCE_FACTOR=$(echo $TASK_CONFIG | jq -r '.scheduler_reduce_factor')
SCHEDULER_REDUCE_PATIENCE=$(echo $TASK_CONFIG | jq -r '.scheduler_reduce_patience')
## Training args.
BATCH_SIZE=$(echo $TASK_CONFIG | jq -r '.batch_size')
MAX_EPOCH=$(echo $TASK_CONFIG | jq -r '.max_epoch')
EVAL_METRIC=$(echo $TASK_CONFIG | jq -r '.eval_metric')
EARLY_STOP=$(echo $TASK_CONFIG | jq -r '.early_stop')
STOP_PATIENCE=$(echo $TASK_CONFIG | jq -r '.stop_patience')
STOP_MIN_DELTA=$(echo $TASK_CONFIG | jq -r '.stop_min_delta')
STOP_MIN_EPOCH=$(echo $TASK_CONFIG | jq -r '.stop_min_epoch')
NUM_WORKERS=$(echo $TASK_CONFIG | jq -r '.num_workers')
RANDOM_STATE=$(echo $TASK_CONFIG | jq -r '.random_state')
## Utility args.
SAVE_CHECKPOINTS=$(echo $TASK_CONFIG | jq -r '.save_checkpoints')
SAVE_BESTMODEL=$(echo $TASK_CONFIG | jq -r '.save_bestmodel')
CHECKPOINT_INTERVAL=$(echo $TASK_CONFIG | jq -r '.checkpoint_interval')
CLEANUP_INTERVAL=$(echo $TASK_CONFIG | jq -r '.cleanup_interval')
VERBOSE_FLAG=$(echo $TASK_CONFIG | jq -r '.verbose_flag')

# --- Construct command ---
CMD=(python src/cnn_train.py
    --config_label "$CONFIG_LABEL"
    --fold "$FOLD"
    --total_folds "$TOTAL_FOLDS"
    --val_ratio "$VAL_RATIO"
    --lmdb_path "$LMDB_PATH"
    --csv_path "$CSV_PATH"
    --dataset_ver "$DATASET_VER"
    --submit_timestamp "$SUBMIT_TIMESTAMP"
    --model_arch "$MODEL_ARCH"
    --optimizer_name "$OPTIMIZER_NAME"
    --opt_learning_rate "$OPT_LEARNING_RATE"
    --opt_weight_decay "$OPT_WEIGHT_DECAY"
    --scheduler_step "$SCHEDULER_STEP"
    --scheduler_gamma "$SCHEDULER_GAMMA"
    --scheduler_reduce_factor "$SCHEDULER_REDUCE_FACTOR"
    --scheduler_reduce_patience "$SCHEDULER_REDUCE_PATIENCE"
    --batch_size "$BATCH_SIZE"
    --max_epoch "$MAX_EPOCH"
    --eval_metric "$EVAL_METRIC"
    --stop_patience "$STOP_PATIENCE"
    --stop_min_delta "$STOP_MIN_DELTA"
    --stop_min_epoch "$STOP_MIN_EPOCH"
    --num_workers "$NUM_WORKERS"
    --random_state "$RANDOM_STATE"
    --checkpoint_interval "$CHECKPOINT_INTERVAL"
    --cleanup_interval "$CLEANUP_INTERVAL"
)

# Optional flags (only added if not null/false)
[[ "$FREEZE_BACKBONE" == "true" ]] && CMD+=(--freeze_backbone)
[[ "$FREEZE_BACKBONE" == "false" ]] && CMD+=(--no_freeze_backbone)

[[ "$EARLY_STOP" == "true" ]] && CMD+=(--early_stop)
[[ "$EARLY_STOP" == "false" ]] && CMD+=(--no_early_stop)

[[ "$SAVE_CHECKPOINTS" == "true" ]] && CMD+=(--save_checkpoints)
[[ "$SAVE_CHECKPOINTS" == "false" ]] && CMD+=(--no_save_checkpoints)

[[ "$VERBOSE_FLAG" == "true" ]] && CMD+=(--verbose_flag)
[[ "$VERBOSE_FLAG" == "false" ]] && CMD+=(--no_verbose_flag)

[[ "$LAYER_INIT" != "null" ]] && CMD+=(--layer_init "$LAYER_INIT")
[[ "$STATE_DICT_FOLDER" != "null" ]] && CMD+=(--state_dict_folder "$STATE_DICT_FOLDER")
[[ "$SCHEDULER_NAME" != "null" ]] && CMD+=(--scheduler_name "$SCHEDULER_NAME")

# --- Run command ---
echo "SLURM array job ID: $SLURM_ARRAY_JOB_ID"
echo "Running task index: $SLURM_ARRAY_TASK_ID"
echo "SLURM job ID: $SLURM_JOB_ID"
echo
echo "Config label: $CONFIG_LABEL"
echo "Fold: $FOLD of $TOTAL_FOLDS"
echo
echo "Running command: ${CMD[*]}"
"${CMD[@]}"
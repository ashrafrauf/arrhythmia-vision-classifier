#!/bin/bash
#SBATCH --job-name=convnextbase_config2_l2                   # Job name for the array                            # <--- ADJUST JOB TITLE !!!
#SBATCH --nodes=1                                           # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                                 # Run one task
#SBATCH --cpus-per-task=6                                   # CPUs per task (adjust based on cnn_train.py's num_workers)
#SBATCH --gres=gpu:1                               # Request 1 GPU per task (adjust or remove if not using GPUs)
#SBATCH --mem=36G                                           # Memory per task (adjust)
#SBATCH --time=0-48:00:00                                   # Max time for each task hrs:min:sec
#SBATCH --partition=gengpu                              # Your SLURM partition
#SBATCH --constraint=cuda12
#SBATCH --mail-type=ALL                                     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=username@citystgeorges.ac.uk    # Where to send mail
#SBATCH --output=/users/userid/ecg-cnn-home/final-results/convnextbase_config2_adj_l2/slurm-logs/out/slurm-%j.out   # <--- UPDATE EXPERIMENT FOLDER !!!
#SBATCH --error=/users/userid/ecg-cnn-home/final-results/convnextbase_config2_adj_l2/slurm-logs/err/slurm-%j.err    # <--- UPDATE EXPERIMENT FOLDER !!!


# -- Context ---
# Orchestrates the consolidation of multiple grid search experiments within City University's Hyperion.
# Script was built based on examples here: https://cityuni.service-now.com/sp?id=kb_article_view&sysparm_article=KB0012621


# --- Define Base Paths ---
PROJECT_ROOT="/users/userid/ecg-cnn-home"


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

# --- Construct command ---
CMD=(python src/consol_results.py
    --model_arch "${MODEL_ARCH}"
    --dataset_config "${DATASET_CONFIG}"
    --dataset_config_suffix "${CONFIG_SUFFIX}"
    --train_mode "${TRAIN_MODE:-full}"
    --label_col "${LABEL_COL:-rhythm_l1_enc}"
    --runtime_env slurm
)

# --- Run command ---
echo "SLURM job ID: $SLURM_JOB_ID"
echo
echo "Running command: ${CMD[*]}"
"${CMD[@]}"
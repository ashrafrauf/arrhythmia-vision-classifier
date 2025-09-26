#!/bin/bash
#SBATCH --job-name=best_mobilesmall_full_config1                  # Job name for the array
#SBATCH --nodes=1                                           # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                                 # Run one task
#SBATCH --cpus-per-task=6                                   # CPUs per task (adjust based on cnn_train.py's num_workers)
#SBATCH --gres=gpu:1                                        # Request 1 GPU per task (adjust or remove if not using GPUs)
#SBATCH --mem=36G                                           # Memory per task (adjust)
#SBATCH --time=0-24:00:00                                   # Max time for each task hrs:min:sec
#SBATCH --partition=gengpu                                  # Your SLURM partition
#SBATCH --constraint=cuda12
#SBATCH --mail-type=ALL                                     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=username@citystgeorges.ac.uk    # Where to send mail
#SBATCH --output=/users/userid/ecg-cnn-home/experiments/250830_0214_mobilenetv3small_full_config1/slurm-logs/out/slurm-%j.out   # <--- UPDATE EXPERIMENT FOLDER !!!
#SBATCH --error=/users/userid/ecg-cnn-home/experiments/250830_0214_mobilenetv3small_full_config1/slurm-logs/err/slurm-%j.err    # <--- UPDATE EXPERIMENT FOLDER !!!


# -- Context ---
# Orchestrates the consolidation of a particular grid search results within City University's Hyperion.
# Script was built based on examples here: https://cityuni.service-now.com/sp?id=kb_article_view&sysparm_article=KB0012621


# --- Define Base Paths ---
PROJECT_ROOT="/users/userid/ecg-cnn-home"
EXPERIMENT_FOLDER="250830_0214_mobilenetv3small_full_config1"                                                                       # <--- UPDATE EXPERIMENT FOLDER !!!


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
CMD=(python src/cnn_train_best_model.py
    --experiment_folder "$EXPERIMENT_FOLDER"
    --train_env slurm
)


# --- Run command ---
echo "SLURM job ID: $SLURM_JOB_ID"
echo
echo "Running command: ${CMD[*]}"
"${CMD[@]}"
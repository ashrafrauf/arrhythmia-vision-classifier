# Grid Search Folder

## Introduction
This directory contains the materials (input and output) of the grid search optimisation conducted. Each folder is a grid search for a specific model and dataset config, and at a specific training stage. The folder naming convention follows `{YYMMDD}_{HHMM}_{model}_{trainingstage}_{datasetconfig}`. The results for all experiments will then be consolidated into a different folder (final-results).

<br/>

## Folder Structure
Illustrative folder structure as shown below. Each folder is a self-contained experiment that explores a specific grid search space.
```
experiments
 ┃
 ┣ 250705_2105_resnet18_head_config1
 ┃ ┣ configs
 ┃ ┃ ┣ best_model_config.json
 ┃ ┃ ┣ experiment_hyperparameter_grid.json
 ┃ ┃ ┗ slurm_job_array_manifest.json
 ┃ ┣ final-models
 ┃ ┃ ┣ best
 ┃ ┃ ┃ ┗ best_model.pth
 ┃ ┃ ┗ cache
 ┃ ┣ final-results
 ┃ ┃ ┗ best_model_results.json
 ┃ ┣ grid-results
 ┃ ┣ local-logs
 ┃ ┃ ┣ err
 ┃ ┃ ┗ out
 ┃ ┣ slurm-logs
 ┃ ┃ ┣ err
 ┃ ┃ ┗ out
 ┃
 ┣ 250709_0451_resnet18_full_config1
 ┃ ┣ configs
 ┃ ┃ ┣ best_model_config.json
 ┃ ┃ ┣ experiment_hyperparameter_grid.json
 ┃ ┃ ┗ slurm_job_array_manifest.json
 ┃ ┣ final-models
 ┃ ┃ ┣ best
 ┃ ┃ ┃ ┗ best_model.pth
 ┃ ┃ ┗ cache
 ┃ ┣ final-results
 ┃ ┃ ┗ best_model_results.json
 ┃ ┣ grid-results
 ┃ ┣ local-logs
 ┃ ┃ ┣ err
 ┃ ┃ ┗ out
 ┃ ┣ slurm-logs
 ┃ ┃ ┣ err
 ┃ ┃ ┗ out
 ┃
 ┣ 250712_1024_efficientnetb3_head_config1
 ┃ ┣ configs
 ┃ ┃ ┣ best_model_config.json
 ┃ ┃ ┣ experiment_hyperparameter_grid.json
 ┃ ┃ ┗ slurm_job_array_manifest.json
 ┃ ┣ final-models
 ┃ ┃ ┣ best
 ┃ ┃ ┃ ┗ best_model.pth
 ┃ ┃ ┗ cache
 ┃ ┣ final-results
 ┃ ┃ ┗ best_model_results.json
 ┃ ┣ grid-results
 ┃ ┣ local-logs
 ┃ ┃ ┣ err
 ┃ ┃ ┗ out
 ┃ ┣ slurm-logs
 ┃ ┃ ┣ err
 ┃ ┃ ┗ out
```
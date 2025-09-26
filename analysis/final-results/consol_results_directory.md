# Final Results Folder

## Introduction
This directory contains the consolidated results for all the grid search conducted for a specific model and dataset config, across both training stages. The folder naming convention follows `{model}_{datasetconfig}`. Specifically, for each combination of model and dataset config, there exists:
1. Detailed results of grid search at each individual fold level in a csv file.
2. Summarised results of grid search at hyperparameter combination level in a csv file.
3. Best hyperparameter configuration in a json file.
4. State dict of model, retrained using best hyperparameters, in pth format.
5. Training and validation history of the training run using best hyperparameters in a dictionary in a json file.
6. Actual and predicted labels, including the probablity matrix, evaluated on the training and test set, stored in a json file.
7. Calculated evaluation metrics (accuracy, F1, precision, recall, AUROC), stored in a dictionary in a json file.

The evaluation metrics across all models and dataset configs are then consolidated into a single CSV file (analysis_df.csv) for further analysis.

<br/>

## Folder Structure
Illustrative folder structure as shown below.
```
final-results
 ┃
 ┣ resnet18_config1
 ┃ ┣ model-retrain-cache
 ┃ ┣ slurm-logs
 ┃ ┃ ┣ err
 ┃ ┃ ┣ out
 ┃ ┣ best_model_config.json
 ┃ ┣ best_model_retrained.pth
 ┃ ┣ consol_grid_results_detailed.csv
 ┃ ┣ consol_grid_results_summary.csv
 ┃ ┣ model_retrained_eval_metrics_slurm_cuda.json
 ┃ ┣ model_retrained_history.json
 ┃ ┗ model_retrained_predictions.json
 ┃
 ┣ efficientnetb3_config1
 ┃ ┣ model-retrain-cache
 ┃ ┣ slurm-logs
 ┃ ┃ ┣ err
 ┃ ┃ ┣ out
 ┃ ┣ best_model_config.json
 ┃ ┣ best_model_retrained.pth
 ┃ ┣ consol_grid_results_detailed.csv
 ┃ ┣ consol_grid_results_summary.csv
 ┃ ┣ model_retrained_eval_metrics_slurm_cuda.json
 ┃ ┣ model_retrained_history.json
 ┃ ┗ model_retrained_predictions.json
 ┃
 ┗ analysis_df.csv
```
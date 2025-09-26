# Bayesian Search Folder

## Introduction
This directory contains the materials (input and output) of the Bayesian search optimisation conducted. Each folder is a Bayesian search for a specific model and dataset config, only done for the second training stage. The folder naming convention follows `{model}_{trainingstage}_{datasetconfig}`. For each search, the results of each trial are stored in an sqlite database named as optuna_study.db to allow for interrupted runs. Selected evaluation metrics are also stored in grid-results folder for easy access.

<br/>

## Folder Structure
Illustrative folder structure as shown below. Each folder is a Bayesian search for a specific model and dataset config.
```
optuna
 ┃
 ┣ resnet18_full_config1
 ┃ ┣ final-models
 ┃ ┃ ┣ best
 ┃ ┃ ┗ cache
 ┃ ┣ grid-results
 ┃ ┣ local-logs
 ┃ ┃ ┣ err
 ┃ ┃ ┗ out
 ┃ ┣ optuna-results
 ┃ ┃ ┣ best_trial.json
 ┃ ┃ ┣ optuna_study.db
 ┃ ┃ ┗ study_summary.txt
 ┃ ┣ slurm-logs
 ┃ ┃ ┣ err
 ┃ ┃ ┗ out
 ┃
 ┣ efficientnetb3_full_config1
 ┃ ┣ final-models
 ┃ ┃ ┣ best
 ┃ ┃ ┗ cache
 ┃ ┣ grid-results
 ┃ ┣ local-logs
 ┃ ┃ ┣ err
 ┃ ┃ ┗ out
 ┃ ┣ optuna-results
 ┃ ┃ ┣ best_trial.json
 ┃ ┃ ┣ optuna_study.db
 ┃ ┃ ┗ study_summary.txt
 ┃ ┣ slurm-logs
 ┃ ┃ ┣ err
 ┃ ┃ ┗ out

```
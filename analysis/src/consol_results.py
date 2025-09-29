# --- Baseline packages ---
import numpy as np
import pandas as pd

# --- Utility packages ---
import os
import argparse
import json
import joblib
import datetime
import time

# --- Helper functions ---
import ecg_utils
from cnn_train import train_cnn_model


def consolidate_experiments(
        model_arch,
        dataset_config,
        main_data_dir,
        retrain_dataset,
        train_mode = 'full',
        best_metric = 'f1',
        best_agg_method = 'mean',
        hyperparam_list = ['optimizer', 'learning_rate', 'weight_decay', 'scheduler', 'batch_size', 'init_method'],
        model_label = 'best_model_retrained',
        label_col = 'rhythm_l1_enc',
        runtime_env = 'local'
):
    """
    Consolidates results for across multiple experiments that explored different grid
    search spaces for the selected model and dataset config in the second training stage.
    Identifies the best hyperparameter combination across all experiments, then re-runs
    the training using the best hyperparameter combination. Saves the output in the
    final-results directory.
    """
    # Set global seeds.
    ecg_utils.set_global_seeds(seed_num=42)

    # Get runtime.
    device = ecg_utils.check_runtime_device()

    # Identify script location and project directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src folder
    project_dir = os.path.dirname(script_dir)  # analysis folder

    # Identify relevant folders.
    main_res_dir = os.path.join(project_dir, "final-results", f'{model_arch}_{retrain_dataset}')
    cache_dir = os.path.join(main_res_dir, "model-retrain-cache")

    # Get label encoder.
    rhythm_encoder_path = os.path.join(main_data_dir, "metadata", "rhythm_encoders_dict.joblib")
    rhythm_encoders = joblib.load(rhythm_encoder_path)
    l1_rhythm_encoder = rhythm_encoders[label_col]

    # Identify dataset for retraining.
    lmdb_train_path, csv_train_path = ecg_utils.get_dataset_paths(
        main_data_dir,
        model_arch,
        retrain_dataset,
        'train'
    )

    # Consolidate results.
    results_consol_df, results_smry_df = ecg_utils.combine_experiment_results(
        model_arch,
        project_dir,
        dataset_config = dataset_config,
        train_type = train_mode
    )

    # Get best config label.
    best_config_row = ecg_utils.get_best_config(
        results_smry_df,
        bestMetric = best_metric,
        bestAggMethod = best_agg_method,
        hyperparamsList = hyperparam_list
    )

    # Get best configs.
    best_experiment_folder = best_config_row.folder_name
    grid_config_filepath = os.path.join(project_dir, "experiments", best_experiment_folder, "configs", "slurm_job_array_manifest.json")
    with open(grid_config_filepath, 'r') as file:
        grid_config_dict = json.load(file)
    
    best_config_label = best_config_row.model_num
    best_config = [config for config in grid_config_dict if config['config_label']==best_config_label][0]  # Only select the first value, the rest are similar values for diffferent folds.

    # Save grid results and best config.
    grid_res_detailed_filepath = os.path.join(main_res_dir, "consol_grid_results_detailed.csv")
    results_consol_df.to_csv(grid_res_detailed_filepath, index=False)

    grid_res_smry_filepath = os.path.join(main_res_dir, "consol_grid_results_summary.csv")
    results_smry_df.to_csv(grid_res_smry_filepath, index=False)

    best_config_filepath = os.path.join(main_res_dir, "best_model_config.json")
    with open(best_config_filepath, 'w') as f:
        json.dump(best_config, f, indent=4)
    
    # Retrain best model config.
    print(f"[{datetime.datetime.now()}]   Retraining model...")

    # Identify state_dict_path if folder provided.
    if best_config.get('state_dict_folder', None) is not None:
        state_dict_path = os.path.join(project_dir, "experiments", best_config['state_dict_folder'], "final-models", "best", "best_model.pth")
    else:
        state_dict_path = None
    
    # Run training loop. Hardcoded num_workers and minimum train epochs.
    retrain_model, bestMetricsDict, metricsHist = train_cnn_model(
        lmdb_train_path,
        csv_train_path,
        model_arch,
        best_config['optimizer_name'],
        cache_dir,
        main_res_dir,
        batchSize = best_config['batch_size'],
        numWorkers = 2,
        randomState = best_config['random_state'],
        trainIdx = None,
        valIdx = None,
        valSplitRatio = 0.2,
        freezeBackbone = best_config['freeze_backbone'],
        initMethod = best_config['layer_init'],
        stateDictPath = state_dict_path,
        learningRate = best_config['opt_learning_rate'],
        weightDecay = best_config['opt_weight_decay'],
        schedulerName = best_config['scheduler_name'],
        schedulerStepSize = best_config.get('scheduler_step', 20),
        schedulerGamma = best_config.get('scheduler_gamma',0.5),
        schedulerReduceFactor = best_config.get('scheduler_reduce_factor', 0.1),
        schedulerReducePatience = best_config.get('scheduler_reduce_patience', 5),
        evalMetric = best_config['eval_metric'],
        evalPatience = best_config['stop_patience'],
        evalMinDelta = best_config['stop_min_delta'],
        minEpochStop = 50,
        modelLabel = model_label,
        cacheInterval = best_config['checkpoint_interval'],
        verbose = False,
        numEpochs = best_config['max_epoch'],
        saveCheckpoint = True,
        saveBestModel = True,
        earlyStop = best_config['early_stop'],
        cleanupInterval = best_config['cleanup_interval']
    )

    # Save metrics and retraining history.
    result_dict = {
        'best_metrics': bestMetricsDict,
        'best_train_history': metricsHist
    }
    res_filepath = os.path.join(main_res_dir, f'model_retrained_history.json')
    with open(res_filepath, 'w') as f:
        json.dump(result_dict, f, indent=4)
    print(f'\n[{datetime.datetime.now()}]   Training history and validation metrics for best epoch are saved at: {res_filepath}')

    # Get test datasets.
    lmdb_test_path, csv_test_path = ecg_utils.get_dataset_paths(
        main_data_dir,
        model_arch,
        retrain_dataset,
        'test'
    )

    # Evaluate predictions on training set.
    print(f"\n[{datetime.datetime.now()}]   Evaluating model predictions on training set...")
    train_start_time = time.time()
    train_outputs, train_metric_scores_dict = ecg_utils.evaluate_model_predictions(
        lmdb_train_path,
        csv_train_path,
        model = retrain_model,
        label_enc = l1_rhythm_encoder,
        device = device,
        metadata = True
    )
    train_elapsed_time = time.time() - train_start_time
    train_metric_scores_dict['total_inference_time'] = train_elapsed_time
    train_metric_scores_dict['throughput'] = train_metric_scores_dict['n_samples'] / train_elapsed_time
    print(f'Total time taken for inference on training set: {train_elapsed_time // 60:.0f}m {train_elapsed_time % 60:.0f}s!')

    # Evaluate predictions on test set.
    print(f"\n[{datetime.datetime.now()}]   Evaluating model predictions on test set...")
    test_start_time = time.time()
    test_outputs, test_metric_scores_dict = ecg_utils.evaluate_model_predictions(
        lmdb_test_path,
        csv_test_path,
        model = retrain_model,
        label_enc = l1_rhythm_encoder,
        device = device,
        metadata = True
    )
    test_elapsed_time = time.time() - test_start_time
    test_metric_scores_dict['total_inference_time'] = test_elapsed_time
    test_metric_scores_dict['throughput'] = test_metric_scores_dict['n_samples'] / test_elapsed_time
    print(f'Total time taken for inference on test set: {test_elapsed_time // 60:.0f}m {test_elapsed_time % 60:.0f}s!')

    # Save prediction outputs. First, convert numpy arrays to list to make it JSON serializable.
    train_outputs_list = {key: value.tolist() for key, value in train_outputs.items()}
    test_outputs_list = {key: value.tolist() for key, value in test_outputs.items()}
    pred_outputs = {
        'train': train_outputs_list,
        'test': test_outputs_list
    }
    pred_filepath = os.path.join(main_res_dir, f'model_retrained_predictions.json')
    with open(pred_filepath, 'w') as f:
        json.dump(pred_outputs, f, indent=4)
    print(f'\n[{datetime.datetime.now()}]   Model prediction outputs are saved at: {pred_filepath}')

    # Save evaluation results.
    eval_ouputs = {
        'train': train_metric_scores_dict,
        'test': test_metric_scores_dict
    }
    eval_filepath = os.path.join(main_res_dir, f'model_retrained_eval_metrics_{runtime_env}_{device}.json')
    with open(eval_filepath, 'w') as f:
        json.dump(eval_ouputs, f, indent=4)
    print(f'\n[{datetime.datetime.now()}]   Model evaluation metrics are saved at: {eval_filepath}')

    return None





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_arch", type=str, required=True, help="Model architecture used for the experiment.")
    parser.add_argument("--dataset_config", type=str, required=True, help='Dataset config used for the experiment.')
    parser.add_argument("--dataset_config_suffix", type=str, default="None", help='Optional additional suffix used to indicate variations of the selected dataset config.')
    parser.add_argument("--train_mode", type=str, default="full", help="'Head' for first training stage and 'full' for second training stage.")
    parser.add_argument("--best_metric", type=str, default="f1", help='Evaluation metric to be used to select the best hyperparameter configuration.')
    parser.add_argument("--best_agg_method", type=str, default="mean", help='Summary statistic used to select the best hyperparameter configuration.')
    parser.add_argument("--model_label", type=str, default="best_model_retrained", help='Filename for model state dict object.')
    parser.add_argument("--runtime_env", type=str, default="local", help='Runtime environment')
    parser.add_argument("--label_col", type=str, default="rhythm_l1_enc", help='To indicate whether using merged ("rhythm_l1_enc") or granular ("rhythm_l2_enc") labels.')
    args = parser.parse_args()

    # Identify scipt location and project directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src folder
    project_dir = os.path.dirname(script_dir)  # analysis folder
    
    # Identify location of main data directory.
    if args.runtime_env=='local':
        main_data_dir = os.path.join(os.path.dirname(project_dir), 'data')
    elif args.runtime_env=='slurm':
        main_data_dir = '/users/userid/archive/ecg-cnn-data'

    # Identify dataset for retraining.
    if args.dataset_config_suffix == "None":
        retrain_dataset = args.dataset_config
    else:
        retrain_dataset = f'{args.dataset_config}_{args.dataset_config_suffix}'
    
    # Create relevant folders.
    main_res_dir = os.path.join(project_dir, "final-results", f'{args.model_arch}_{retrain_dataset}')
    cache_dir = os.path.join(main_res_dir, "model-retrain-cache")
    slurm_logs_out_dir = os.path.join(main_res_dir, "slurm-logs", "out")
    slurm_logs_err_dir = os.path.join(main_res_dir, "slurm-logs", "err")

    os.makedirs(main_res_dir, exist_ok=True)
    os.makedirs(slurm_logs_out_dir, exist_ok=True)
    os.makedirs(slurm_logs_err_dir, exist_ok=True)
    
    # Run main script.
    consolidate_experiments(
        args.model_arch,
        args.dataset_config,
        main_data_dir,
        retrain_dataset,
        train_mode = args.train_mode,
        best_metric = args.best_metric,
        best_agg_method = args.best_agg_method,
        hyperparam_list = ['optimizer', 'learning_rate', 'weight_decay', 'scheduler', 'batch_size', 'init_method'],
        model_label = args.model_label,
        label_col = args.label_col,
        runtime_env = args.runtime_env
    )
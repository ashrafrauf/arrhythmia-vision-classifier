# --- Baseline packages ---
import pandas as pd

# --- Deep learning packages ---
import torch
import torchinfo
from thop import profile

# --- Utility packages ---
import os
import json
import datetime

# --- Helper functions ---
from .ecg_model_utils import get_cnn_model



def combine_fold_results(foldResultsPath):
    """
    Consolidate the results of the kfold cross-validation implemented during
    the grid search optimisation process. Assumes the results for each fold
    and hyperparameter combination are stored in separate json files within
    a single folder.

    Arguments:
        foldResultsPath (str): Path to the directory where the grid search results are stored.
    
    Returns:
        pd.DataFrame: A DataFrame containing the evaluation metrics for each fold.
        pd.DataFrame: A DataFrame containing the mean and std deviation for each metric across folds for each hyperparameter combination.
    """
    # Consolidate fold-level results for grid search.
    resultsConsolList = []
    for jsonFile in os.listdir(foldResultsPath):
        jsonFilePath = os.path.join(foldResultsPath, jsonFile)
        with open(jsonFilePath, 'r') as f:
            resultsConsolList.append(json.load(f))
    resultsConsolDataframe = pd.DataFrame(resultsConsolList)

    # Summarise results for each config.
    evalMetrics = ['loss', 'accuracy', 'f1', 'precision', 'recall', 'best_epoch', 'time_total_training', 'time_epoch_average']
    resultsSummaryDataframe = resultsConsolDataframe.groupby(
        [col for col in resultsConsolDataframe.columns if col not in evalMetrics and col != "fold_num"],
        dropna=False,
        as_index=False
    ).agg({
        'loss': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'best_epoch': ['mean', 'std']
    })
    resultsSummaryDataframe.columns = [('_').join(col).strip('_') for col in resultsSummaryDataframe.columns]
    
    return resultsConsolDataframe, resultsSummaryDataframe



def get_best_config(
        summaryDataframe,
        bestMetric='f1',
        bestAggMethod='mean',
        hyperparamsList = ['optimizer', 'learning_rate', 'weight_decay', 'scheduler', 'batch_size', 'init_method']
):
    """
    Identifies the best hyperparameter combination from a DataFrame containing the mean
    and standard deviation or each metric across folds for each hyperparameter combination.

    Arguments:
        summaryDataframe (pd.DataFrame): DataFrame containing summarised evaluation metrics per hyperparameter combination.
        bestMetric (str, optional): Evaluation metric to be used to select the best hyperparameter configuration. (Default: 'f1')
        bestAggMethod (str, optional): Summary statistic used to select the best hyperparameter configuration. (Default: 'mean')
        hyperparamsList (list, optional): List of hyperparameters that were optimised, only used for printing summary to consol. (Default: ['optimizer', 'learning_rate', 'weight_decay', 'scheduler', 'batch_size', 'init_method'])
    
    Returns:
        pd.DataFrame: A one-row DataFrame containing the best hyperparameter configuration.
    """
    assert bestMetric in ['loss', 'accuracy', 'precision', 'recall', 'f1'], f"{bestMetric} is not supported. Try a different metric."
    assert bestAggMethod in ['mean', 'std'], f"{bestAggMethod} summary statistic aggregation method is not supported. Choose either mean or std."

    # Select best config based on relevant metric.
    chosenMetric = f"{bestMetric}_{bestAggMethod}"
    print(f'\nBest config chosen best on {chosenMetric} evaluation metric.\n')

    if bestMetric=='loss':
        bestConfigRow = summaryDataframe.loc[summaryDataframe[chosenMetric].idxmin()].copy()
    else:
        bestConfigRow = summaryDataframe.loc[summaryDataframe[chosenMetric].idxmax()].copy()
    
    if hasattr(bestConfigRow, 'folder_name'):
        print(f'Best config: {bestConfigRow.model_num} in {bestConfigRow.folder_name}')
    else:
        print(f'Best config: {bestConfigRow.model_num}')
    print(f'Hyperparameter choice:\n{bestConfigRow[hyperparamsList]}')

    return bestConfigRow



def combine_experiment_results(
        model_arch,
        project_dir,
        train_type = 'full',
        dataset_config = 'config1'
):
    """
    Consolidate the results of the kfold cross-validation implemented during the grid
    search optimisation process across different folders. Useful when multiple experiments
    were run for the same model and dataset config. Assumes each folder has the naming
    convention {YYMMDD}_{HHMM}_{model}_{traintype}_{datasetconfig}.

    Arguments:
        model_arch (str): Name of model.
        project_dir (str): Path to analysis directory within the project folder.
        train_type (str, optional): 'head' refers to the first training stage, while 'full' refers to the second training stage. (Default: 'full')
        dataset_config (str, optional): THe dataset config used as inputs. (Default: 'config1')
    
    Returns:
        pd.DataFrame: A DataFrame containing the evaluation metrics for each fold.
        pd.DataFrame: A DataFrame containing the mean and std deviation for each metric across folds for each hyperparameter combination.
    """
    # Identify relevant folders.
    print(f"[{datetime.datetime.now()}]   Identifying relevant folders...")
    experiment_folder_list = [exp_folder for exp_folder in sorted(os.listdir(os.path.join(project_dir, "experiments"))) if not exp_folder.startswith(".")]
    chosen_folders = [f for f in experiment_folder_list if (f.split("_")[2]==model_arch and f.split("_")[3]==train_type and f.split("_")[-1]==dataset_config)]
    print(f"[{datetime.datetime.now()}]   Relevant folders:")
    print(*chosen_folders, sep='  |  ')
    print("\n")

    # Create DataFrame of fold-level results.
    print(f"[{datetime.datetime.now()}]   Consolidating results...")
    consol_results = []
    for exp_folder in chosen_folders:
        print(f"[{datetime.datetime.now()}]   Processing folder {exp_folder}")
        path_to_results = os.path.join(project_dir, "experiments", exp_folder, "grid-results")
        grid_res_list = [file for file in sorted(os.listdir(path_to_results)) if not file.endswith(".DS_Store")]
        run_count = len(grid_res_list)
        print(f"[{datetime.datetime.now()}]   Folder {exp_folder} has {run_count} total runs ({int(run_count/5)} configurations with 5-fold cross-validation.)")
        
        for json_file in grid_res_list:
            json_filepath = os.path.join(path_to_results, json_file)
            with open(json_filepath, 'r') as f:
                grid_res = json.load(f)
            grid_res['folder_name'] = exp_folder
            consol_results.append(grid_res)
    results_full_df = pd.DataFrame(consol_results)
    print(f"[{datetime.datetime.now()}]   Done consolidating results!")

    # Create dataframe of summarised results for each config.
    print(f"[{datetime.datetime.now()}]   Summarising results...")
    eval_metrics = ['loss', 'accuracy', 'f1', 'precision', 'recall', 'best_epoch', 'time_total_training', 'time_epoch_average']
    results_smry_df = results_full_df.groupby(
        [col for col in results_full_df.columns if col not in eval_metrics and col != "fold_num"],
        dropna=False,
        as_index=False
    ).agg({
        'loss': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'best_epoch': ['mean', 'std'],
        'time_total_training': ['mean', 'std'],
        'time_epoch_average': ['mean', 'std']
    })
    results_smry_df.columns = [('_').join(col).strip('_') for col in results_smry_df.columns]
    print(f"[{datetime.datetime.now()}]   Done summarising results!")

    return results_full_df, results_smry_df



def get_eval_metrics(
        final_results_dir,
        metrics_filename = 'model_retrained_eval_metrics_slurm_cuda'
):
    """
    Consolidate evaluation metrics of the main experiment (i.e., inter-architecture comparison of
    predicting four arhhythmia classes using Config1 and Config2 as inputs). Assumes each folder
    within the final_results_dir contains the results for a specific model and dataset config, and
    the results stored within metrics_filename in a dictionary.

    Arguments:
        final_results_dir (str): Path to directory where all the final results are stored.
        metrics_filename (str): Name of file containing the evaluation metrics without the extension. File is assumed to be in json format.
    
    Returns:
        pd.DataFrame: A DataFrame with the results across all experiments consolidated.
    """
    folder_list = [folder_name for folder_name in sorted(os.listdir(final_results_dir)) if not folder_name.startswith('.') if not folder_name.endswith((".csv", "_l2", ".md"))]
    consolidated_final_results = []

    for folder_name in folder_list:
        model_arch = folder_name.split("_")[0]
        dataset_config = folder_name.split("_")[1]

        # Get results.
        metrics_file = os.path.join(final_results_dir, folder_name, f'{metrics_filename}.json')
        with open(metrics_file, 'r') as file:
            metrics_contents = json.load(file)
        eval_env = metrics_filename.split("_")[-2]
        eval_gpu = metrics_filename.split("_")[-1]

        for dataset_type, metric_type in metrics_contents.items():
            for metric_name, values in metric_type.items():
                if isinstance(values, dict):
                    for label_name, label_res in values.items():
                        row = {
                            'model_arch': model_arch,
                            'dataset_config': dataset_config,
                            'dataset_type': dataset_type,
                            'metric_name': metric_name,
                            'label_name': label_name,
                            'results': label_res,
                            'eval_env': eval_env,
                            'eval_gpu': eval_gpu
                        }
                        consolidated_final_results.append(row)
                else:
                    row = {
                        'model_arch': model_arch,
                        'dataset_config': dataset_config,
                        'dataset_type': dataset_type,
                        'metric_name': metric_name,
                        'label_name': 'macro_avg',
                        'results': values,
                        'eval_env': eval_env,
                        'eval_gpu': eval_gpu
                    }
                    consolidated_final_results.append(row)
        
        # Training time stored in a separate json.
        history_filename = 'model_retrained_history'
        history_file = os.path.join(final_results_dir, folder_name, f'{history_filename}.json')
        with open(history_file, 'r') as file:
            history_contents = json.load(file)
        
        train_time_label_dict = {'timeTotalTraining': 'total_train_time', 'timeAveragePerEpoch': 'per_epoch_train_time'}
        for metric in ['timeTotalTraining', 'timeAveragePerEpoch']:
            row = {
                'model_arch': model_arch,
                'dataset_config': dataset_config,
                'dataset_type': 'train',
                'metric_name': train_time_label_dict[metric],
                'label_name': 'macro_avg',
                'results': history_contents['best_metrics'][metric],
                'eval_env': eval_env,
                'eval_gpu': eval_gpu
            }
            consolidated_final_results.append(row)
    
    final_results_df = pd.DataFrame(consolidated_final_results)
    return final_results_df



def get_model_info(
        final_results_dir,
        device,
        num_labels = 4,
        model_filename = 'best_model_retrained'
):
    """
    Extracts key info of the models used in the main experiment (i.e., inter-architecture comparison of
    predicting four arhhythmia classes using Config1 and Config2 as inputs). Assumes each folder
    within the final_results_dir contains the results for a specific model and dataset config. MACs are
    obtained using thop library, while the other info are obtained using torchinfo library. Note: the MACs
    obtained using torchinfo seemed to be inaccurate for ConvNeXt models.

    Arguments:
        final_results_dir (str): Path to directory where all the final results are stored.
        device (str): Runtime environment.
        num_labels (int): Number of unique classes, used to replace the classification head.
        model_filename (str): Filename of the model state dict, without the extension. Assumed to be saved as .pth format.
    
    Returns:
        pd.DataFrame: DataFrame containing the number of parameters, MACs, model disk size, and parameter size for all the models.
        dict: A dictionary containing torchinfo.summary object for each of the model.
    """
    folder_list = [folder_name for folder_name in sorted(os.listdir(final_results_dir)) if not folder_name.startswith('.') if not folder_name.endswith((".csv", "_l2", ".md"))]
    consolidated_model_info = []
    model_stats_class = {}

    for folder_name in folder_list:
        model_arch = folder_name.split("_")[0]

        # Get model info.
        state_dict_path = os.path.join(final_results_dir, folder_name, f'{model_filename}.pth')
        cnn_model = get_cnn_model(model_arch, num_labels, freezeBackbone=False, stateDictPath=state_dict_path)
        if model_arch == 'efficientnetb3':
            # Use torchinfo for most metrics...
            model_stats_info = torchinfo.summary(cnn_model, input_size=(1, 3, 300, 300), verbose=0, device=device,
                                                 col_names = ["input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"])
            # ... except for MACs, use thop. torchinfo MAC calculation for convnextbase is inaccurate.
            # See GitHub issue: https://github.com/TylerYep/torchinfo/issues/126
            # Also see: https://github.com/TylerYep/torchinfo/issues/312
            input = torch.randn(1, 3, 300, 300).to(device)
            macs, _ = profile(cnn_model, inputs=(input, ))
        else:
            # Use torchinfo for most metrics...
            model_stats_info = torchinfo.summary(cnn_model, input_size=(1, 3, 224, 224), verbose=0, device=device,
                                                 col_names = ["input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"])
            # ... except for MACs, use thop. torchinfo MAC calculation for convnextbase is inaccurate.
            input = torch.randn(1, 3, 224, 224).to(device)
            macs, _ = profile(cnn_model, inputs=(input, ))
        
        total_params = model_stats_info.total_params
        mult_add_gb_wrong = model_stats_info.to_readable(model_stats_info.total_mult_adds, 'G')[1]
        mult_add_gb = macs / 1e9
        input_size_mb = model_stats_info.to_readable(model_stats_info.total_input, 'M')[1]
        pass_size_mb = model_stats_info.to_readable(model_stats_info.total_output_bytes, 'M')[1]
        param_size_mb = model_stats_info.to_readable(model_stats_info.total_param_bytes, 'M')[1]
        total_size_mb = model_stats_info.to_readable(model_stats_info.total_output_bytes + model_stats_info.total_input + model_stats_info.total_param_bytes, 'M')[1]

        model_info = {
            'model_arch': model_arch,
            'total_params': total_params,
            'mac_gb': mult_add_gb,
            'input_size_mb': input_size_mb,
            'pass_size_mb': pass_size_mb,
            'param_size_mb': param_size_mb,
            'total_size_mb': total_size_mb,
            'mac_gb_wrong': mult_add_gb_wrong
        }
        consolidated_model_info.append(model_info)
        model_stats_class[model_arch] = model_stats_info
    
    model_info_df = pd.DataFrame(consolidated_model_info).drop_duplicates().reset_index(drop=True)
    return model_info_df, model_stats_class
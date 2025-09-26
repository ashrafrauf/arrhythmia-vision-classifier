# --- Utility packages ---
import os
import datetime
import time
import json
import joblib
import argparse

# --- Helper functions ---
import ecg_utils
from cnn_train import train_cnn_model



def main(
        model_arch,
        main_data_dir,
        retrain_dataset,
        seed_offset = 0,
        max_seed = 25,
        label_col = 'rhythm_l1_enc',
        res_folder = 'seed-runs-mod',
        runtime_env = 'slurm'
):
    """
    Orchestrator script to implement multiple training runs with different intialised
    random seeds. Random seeds are selected such that to be multiple of the original
    seed, up to a total of max_seed.
    """
    # Set global seeds.
    ecg_utils.set_global_seeds(seed_num=42)

    # Get runtime.
    device = ecg_utils.check_runtime_device()

    # Identify scipt location and project directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src folder
    project_dir = os.path.dirname(script_dir)  # analysis folder

    # Identify relevant folders.
    print(f'\n[{datetime.datetime.now()}]   Seed experiments for {model_arch} with {retrain_dataset}.')
    main_res_dir = os.path.join(project_dir, "final-results", f'{model_arch}_{retrain_dataset}')
    cache_dir = os.path.join(main_res_dir, "model-retrain-cache")

    # Create folder to store results.
    history_dir = os.path.join(main_res_dir, res_folder, "history-results")
    eval_dir = os.path.join(main_res_dir, res_folder, "eval-results")
    os.makedirs(history_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Get train datasets.
    lmdb_train_path, csv_train_path = ecg_utils.get_dataset_paths(
        main_data_dir,
        model_arch,
        retrain_dataset,
        'train'
    )

    # Get test datasets.
    lmdb_test_path, csv_test_path = ecg_utils.get_dataset_paths(
        main_data_dir,
        model_arch,
        retrain_dataset,
        'test'
    )

    # Get label encoder.
    rhythm_encoder_path = os.path.join(main_data_dir, "metadata", "rhythm_encoders_dict.joblib")
    rhythm_encoders = joblib.load(rhythm_encoder_path)
    l1_rhythm_encoder = rhythm_encoders[label_col]

    # Get best config.
    best_config_filepath = os.path.join(main_res_dir, "best_model_config.json")
    with open(best_config_filepath, 'r') as file:
        best_config = json.load(file)
    
    # Different random seed experiments.
    for i in range(seed_offset, max_seed + seed_offset):
        random_seed_run = best_config['random_state'] * (i + 1)
        model_label = f'model_retrained_seed_{random_seed_run:04d}'
        print(f'\n[{datetime.datetime.now()}]   Running for random seed: {random_seed_run}')
        
        # Run training loop. Hardcoded num_workers and minimum train epochs.
        # Train both classifier head and CNN backbone at the same time, unlike main experiment (i.e. unfreeze backbone but no state_dict_path).
        seeded_model, bestMetricsDict, metricsHist = train_cnn_model(
            lmdb_train_path,
            csv_train_path,
            model_arch,
            best_config['optimizer_name'],
            cache_dir,
            main_res_dir,
            batchSize = best_config['batch_size'],
            numWorkers = 2,
            randomState = random_seed_run,
            trainIdx = None,
            valIdx = None,
            valSplitRatio = 0.2,
            freezeBackbone = False,
            initMethod = best_config['layer_init'],
            stateDictPath = None,
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
            minEpochStop = 30,
            modelLabel = model_label,
            cacheInterval = best_config['checkpoint_interval'],
            verbose = False,
            numEpochs = best_config['max_epoch'],
            saveCheckpoint = False,
            saveBestModel = False,
            earlyStop = best_config['early_stop'],
            cleanupInterval = best_config['cleanup_interval']
        )

        # Save metrics and retraining history.
        result_dict = {
            'best_metrics': bestMetricsDict,
            'best_train_history': metricsHist
        }
        res_filepath = os.path.join(history_dir, f'{model_label}_history.json')
        with open(res_filepath, 'w') as f:
            json.dump(result_dict, f, indent=4)
        print(f'\n[{datetime.datetime.now()}]   Training history and validation metrics for seed {random_seed_run} are saved at: {res_filepath}')

        # Evaluate predictions on training set.
        print(f"\n[{datetime.datetime.now()}]   Evaluating model predictions on training set...")
        train_start_time = time.time()
        train_outputs, train_metric_scores_dict = ecg_utils.evaluate_model_predictions(
            lmdb_train_path,
            csv_train_path,
            model = seeded_model,
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
            model = seeded_model,
            label_enc = l1_rhythm_encoder,
            device = device,
            metadata = True
        )
        test_elapsed_time = time.time() - test_start_time
        test_metric_scores_dict['total_inference_time'] = test_elapsed_time
        test_metric_scores_dict['throughput'] = test_metric_scores_dict['n_samples'] / test_elapsed_time
        print(f'Total time taken for inference on test set: {test_elapsed_time // 60:.0f}m {test_elapsed_time % 60:.0f}s!')

        # Save evaluation results.
        eval_ouputs = {
            'train': train_metric_scores_dict,
            'test': test_metric_scores_dict
        }
        eval_filepath = os.path.join(eval_dir, f'{model_label}_eval_{runtime_env}_{device}.json')
        with open(eval_filepath, 'w') as f:
            json.dump(eval_ouputs, f, indent=4)
        print(f'\n[{datetime.datetime.now()}]   Model evaluation metrics for seed {random_seed_run} are saved at: {eval_filepath}')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_arch", type=str, required=True, help="Model architecture used for the experiment.")
    parser.add_argument("--dataset_config", type=str, required=True, help='Dataset config used for the experiment.')
    parser.add_argument("--dataset_config_suffix", type=str, default="None", help='Optional additional suffix used to indicate variations of the selected dataset config.')
    parser.add_argument("--seed_offset", type=int, default=0, help='Offset to start the multiplicative factor, to be used to break up the training into multiple runs or in case of interrupted runs.')
    parser.add_argument("--max_seed", type=int, default=25, help='Maximum number of seeds to intialise.')
    parser.add_argument("--runtime_env", type=str, default="slurm", help='Runtime environment')
    parser.add_argument("--label_col", type=str, default="rhythm_l1_enc", help='To indicate whether using merged ("rhythm_l1_enc") or granular ("rhythm_l2_enc") labels.')
    parser.add_argument("--res_folder", type=str, default="seed-runs-mod", help='Name of folder to store results.')
    args = parser.parse_args()

    # Identify scipt location and project directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src folder
    project_dir = os.path.dirname(script_dir)  # ecg-cnn-home folder
    
    # Identify location of main data directory.
    if args.runtime_env=='local':
        main_data_dir = '/Users/ashrafrauf/Documents/GitHub/arrhythmia-vision-classifier/data'
    elif args.runtime_env=='slurm':
        main_data_dir = '/users/userid/archive/ecg-cnn-data'

    # Identify dataset for retraining.
    if args.dataset_config_suffix == "None":
        retrain_dataset = args.dataset_config
    else:
        retrain_dataset = f'{args.dataset_config}_{args.dataset_config_suffix}'
    
    # Run main script.
    main(
        args.model_arch,
        main_data_dir,
        retrain_dataset,
        seed_offset = args.seed_offset,
        max_seed = args.max_seed,
        label_col = args.label_col,
        res_folder = args.res_folder,
        runtime_env = args.runtime_env
    )
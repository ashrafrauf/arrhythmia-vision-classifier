# --- Baseline packages ---
import numpy as np
import pandas as pd

# --- Deep learning packages ---
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

# --- Utility packages ---
import os
import time
import datetime
import argparse
import json
import math

# --- Helper functions ---
import ecg_utils

# -- References ---
# This script is a modified and more detailed version of the training script used in the INM460: Computer Vision assignment.
# The general structure is the same - two different loops for training and validation runs.
# However, substantial elements have been modified to suit this project, including making it more modular.



def train_cnn_model(
        datasetLMDBPath,
        metadataCSVPath,
        modelName,
        optimizerName,
        cacheDir,
        bestDir,
        batchSize=10,
        numWorkers=0,
        randomState=42,
        trainIdx=None,
        valIdx=None,
        valSplitRatio=0.2,
        freezeBackbone=True,
        initMethod=None,
        stateDictPath=None,
        learningRate=0.01,
        weightDecay=0.001,
        momentum=0.9,
        schedulerName=None,
        schedulerStepSize=20,       # Only used when scheduler is StepLR.
        schedulerGamma=0.5,         # Only used when scheduler is StepLR.
        schedulerReduceFactor=0.1,  # Only used when scheduler is ReduceLROnPlateau
        schedulerReducePatience=5,  # Only used when scheduler is ReduceLROnPlateau
        evalMetric='f1',
        evalPatience=10,
        evalMinDelta=0,
        minEpochStop=25,
        modelLabel='model_prototype',
        cacheInterval=20,
        verbose=False,
        numEpochs=25,
        saveCheckpoint=False,
        saveBestModel=False,
        earlyStop=False,
        cleanupInterval=0
):
    """
    Trains a PyTorch vision model. First, the dataset is divided into train and validation sets, either via
    given indices or divided internally within the function. Then it train on the training set until
    the maximum number of epochs is reached or the early stopping mechanism is triggered when evaluating
    on the validation set.
    """
    # ------------- #
    # Preliminaries #
    # ------------- #
    # Identify runtime mode.
    device = ecg_utils.check_runtime_device()

    # Start timer to train the model.
    timeStart = time.time()

    # Evaluation mode dictionary.
    metricModeDict = {'loss':'min', 'accuracy':'max', 'precision':'max', 'recall':'max', 'f1':'max'}


    # --------- #
    # Load Data #
    # --------- #
    print("\nLMDB dataset path:", datasetLMDBPath)
    print("Associated csv path:", metadataCSVPath, "\n")

    # Define a custom loader function.
    loader_builder = ecg_utils.CustomSplitDataloader(
        datasetLMDBPath,
        metadataCSVPath,
        batchSize,
        numWorkers,
        randomState
    )

    splitDataLoader, splitDataSizes = loader_builder.build_loaders(
        trainIdx,
        valIdx,
        valSplitRatio,
        returnSplitSize=True
    )


    # ------------------- #
    # Define Architecture #
    # ------------------- #
    # CNN Model
    numClasses = loader_builder.get_num_classes()
    model = ecg_utils.get_cnn_model(
        modelName,
        numClasses,
        freezeBackbone=freezeBackbone,
        initMethod=initMethod,
        stateDictPath=stateDictPath
    )
    model = model.to(device)

    # Loss function.
    model_criterion = nn.CrossEntropyLoss()

    # Optimizer.
    config_optimizer_fn = ecg_utils.make_optimizer_fn(
        optimizerName,
        learningRate,
        weightDecay,
        momentum=momentum
    )
    model_optimizer = config_optimizer_fn(model)

    # Scheduler.
    config_scheduler_fn = ecg_utils.make_scheduler_fn(
        schedulerName,
        stepSize=schedulerStepSize, gammaSize=schedulerGamma,
        metricMode=metricModeDict[evalMetric], reduceFactor=schedulerReduceFactor, reducePatience=schedulerReducePatience,
        epochs=numEpochs, max_lr=learningRate, steps_per_epoch=math.ceil(splitDataSizes['train'] / batchSize), pct_start=0.3, cycle_momentum=True
    )
    model_scheduler = config_scheduler_fn(model_optimizer)


    # ------------------ #
    # Training Utilities #
    # ------------------ #
    # Define metric monitoring.
    modelMetricMonitor = ecg_utils.MetricMonitor(
        metricType=evalMetric,
        metricMode=metricModeDict[evalMetric],
        patience=evalPatience,
        minDelta=evalMinDelta,
        minEpoch=minEpochStop
    )

    # Define checkpoint to enhance fault tolerance during training.
    modelCheckpoint = ecg_utils.CheckpointManager(
        model,
        cacheDir,
        bestDir,
        modelLabel=modelLabel,
        saveInterval=cacheInterval
    )
    
    # Instantiate memory cleanup helper.
    memoryCleanupHelper = ecg_utils.CleanupManager(verbose=verbose)

    # Instantiate variables to store history.
    trainHist = {'loss':[], 'accuracy':[], 'precision':[], 'recall':[], 'f1':[]}
    valHist = {'loss':[], 'accuracy':[], 'precision':[], 'recall':[], 'f1':[]}
    metricsHist = {'train':trainHist, 'val':valHist}



    # ------------------ #
    # Main Training Loop #
    # ------------------ #
    for epoch in range(numEpochs):
        epochTimeStart = time.time()
        print(f"Epoch {epoch}/{numEpochs-1}    Timestamp: {datetime.datetime.now()}", flush=True)
        if verbose:
            print('----------------')

        # Each epoch has a training and validation phase. Set mode accordingly.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Get size of dataset.
            datasetSize = splitDataSizes[phase]

            # Instantiate variables to store total loss and total correct predictions for each epoch.
            runningLoss = 0.0
            runningCorrectPreds = 0

            # Instantiate empty list to store actual and predicted labels for evaluation for each epoch.
            labelTruths = []
            labelPreds = []

            # Iterate over data batches.
            for inputs, labels in splitDataLoader[phase]:
                # Move data to GPU for each batch, if available. Otherwise, remain in cpu.
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Clear the gradients for each batch.
                model_optimizer.zero_grad()

                # Initiate forward pass for each batch. Track gradients only in training phase (i.e., when phase=='train' is True).
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = model_criterion(outputs, labels)

                    # Intiate backward pass and optimize for each batch only in training phase.
                    if phase=='train':
                        loss.backward()
                        model_optimizer.step()
                        if (schedulerName == 'OneCycleLR'):
                            model_scheduler.step()
                
                # Update total loss and total correct predictions for each batch.
                runningLoss += loss.item() * inputs.size(0)
                runningCorrectPreds += torch.sum(preds == labels.data)

                # Store ground truth and predicted labels for each batch.
                labelTruths += labels.tolist()
                labelPreds += preds.tolist()
            
            # Calculate total average loss and accuracy rate for each epoch.
            epochMetricsDict = {
                'epoch': epoch,
                'loss': runningLoss / datasetSize,
                'accuracy': accuracy_score(labelTruths, labelPreds, normalize=True),
                'precision': precision_score(labelTruths, labelPreds, average='macro', zero_division=0),
                'recall': recall_score(labelTruths, labelPreds, average='macro', zero_division=0),
                'f1': f1_score(labelTruths, labelPreds, average='macro', zero_division=0)
            }

            # Print the metrics for each epoch.
            if verbose:
                epochMetricStr = " | ".join([f"{k.capitalize()}: {v:.4f}" for k, v in epochMetricsDict.items() if k != 'epoch'])
                print(f"{phase.upper()} (n={datasetSize})     {epochMetricStr}")
            
            # Store history of evaluation metrics.
            metricsHist[phase]['loss'].append(epochMetricsDict['loss'])
            metricsHist[phase]['accuracy'].append(epochMetricsDict['accuracy'])
            metricsHist[phase]['precision'].append(epochMetricsDict['precision'])
            metricsHist[phase]['recall'].append(epochMetricsDict['recall'])
            metricsHist[phase]['f1'].append(epochMetricsDict['f1'])

            # Update learning rate with scheduler for each epoch.
            if (phase=='val') and (model_scheduler is not None):
                if schedulerName=="ReduceLROnPlateau":
                    model_scheduler.step(epochMetricsDict[evalMetric])
                elif schedulerName != 'OneCycleLR':
                    model_scheduler.step()

            # Check for best best metric and early stopping conditions during validation phase.
            if phase=='val':
                modelMetricMonitor(epoch, epochMetricsDict)

                # Check for best epoch.
                checkBestEpoch = modelMetricMonitor.get_best_epoch()==epoch

                # Checkpoint the model.
                modelCheckpoint.check_and_save(epoch, save_checkpoint=saveCheckpoint, save_bestmodel=saveBestModel, is_best_epoch=checkBestEpoch)

        # End training for one epoch of training and validation.
        epochTimeEnd = time.time()
        epochTimeElapsed = epochTimeEnd - epochTimeStart
        if verbose:
            print(f"Time taken: {epochTimeElapsed // 60:.0f}m {epochTimeElapsed % 60:.0f}s!")
            print("\n")

        # Clean up memory periodically based on specified epochs.
        if (cleanupInterval > 0) and ((epoch + 1) % cleanupInterval == 0):
            memoryCleanupHelper.cleanup_memory()
            print(f"\n--- <<< Memory cleanup done after epoch {epoch} >>> ---\n")
        
        # Check for early stopping conditions if required.
        if earlyStop and modelMetricMonitor.earlyStop:
            print(f"[UPDATE!] Early stopping triggered at epoch {epoch}.\n")
            break
    
    # End training. Store total training time.
    timeElapsed = time.time() - timeStart
    avgTime = timeElapsed / (epoch+1)

    # Get best validation results.
    bestMetricsDict = modelMetricMonitor.get_best_metrics_dict()
    bestMetricsDict['timeTotalTraining'] = timeElapsed
    bestMetricsDict['timeAveragePerEpoch'] = avgTime

    bestEpoch = modelMetricMonitor.get_best_epoch()
    bestModelParams = modelCheckpoint.get_best_params()
    
    print('\n---------------------------------')
    print('---------------------------------\n')
    print(f"Training complete in {timeElapsed // 60:.0f}m {timeElapsed % 60:.0f}s! Average time taken per epoch: {(timeElapsed/(epoch+1)) // 60:.0f}m {(timeElapsed/(epoch+1)) % 60:.0f}.")
    print(f"Best epoch: {bestEpoch} out of {epoch} epochs. Evaluation Metrics:")
    print(f"Loss: {bestMetricsDict['loss']:.4f} | Accuracy: {bestMetricsDict['accuracy']:.4f} | Precision: {bestMetricsDict['precision']:.4f} | Recall: {bestMetricsDict['recall']:.4f} | F1: {bestMetricsDict['f1']:.4f}")
    print('\n---------------------------------')
    print('---------------------------------\n')

    # Load best model parameters and return best model throughout training epochs.
    model.load_state_dict(bestModelParams)
    
    return model, bestMetricsDict, metricsHist





#-------------- #
# Main Function #
#-------------- #
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Preliminaries args.
    parser.add_argument("--config_label", type=str, default="model_prototype", help='Unique label for this model or set of hyperparameter combination.')
    parser.add_argument("--fold", type=int, default=None, help='Fold identifier.')
    parser.add_argument("--total_folds", type=int, default=None, help='Total number of folds.')
    parser.add_argument("--val_ratio", type=float, default=0.2, help='Proportion of training set to be used as validation set.')
    parser.add_argument("--lmdb_path", type=str, required=True, help='Path to LMDB dataset.')
    parser.add_argument("--csv_path", type=str, required=True, help='Path to CSV file with "key" column.')
    parser.add_argument("--dataset_ver", type=str, default="na", help="Dataset version used to train the model.")
    parser.add_argument("--submit_timestamp", type=str, default="na", help="Timestamp when the SLURM job was submitted (YYYYMMDD_HHMM).")

    # Architecture args.
    parser.add_argument("--model_arch", type=str, default="resnet18", help="Model architecture used for the experiment.")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freezes all layers except classification head.")
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false", help="Do not freeze any layers.")
    parser.add_argument("--state_dict_folder", type=str, default=None, help="Path to state dict to load previously trained weights.")
    parser.add_argument("--layer_init", type=str, default="None", help='Weight initialisation method for classifier head. If "None", use the default.')
    parser.add_argument("--optimizer_name", type=str, required=True, help='Optimizer name to be used in the training run.')
    parser.add_argument("--opt_learning_rate", type=float, default=0.01, help='Learning rate to be used in the training run.')
    parser.add_argument("--opt_weight_decay", type=float, default=0.001, help='Weight decay to be used in the training run.')
    parser.add_argument("--scheduler_name", type=str, default="None", help='Scheduler name to be used in the training run.')
    parser.add_argument("--scheduler_step", type=int, default=20, help="Only active when scheduler is StepLR. Number of epochs for each LR decay.")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5, help="Only active when scheduler is StepLR. Multiplicative factor of learning rate decay.")
    parser.add_argument("--scheduler_reduce_factor", type=float, default=0.1, help="Only active when scheduler is ReduceLROnPlateau. Factor by which the learning rate will be reduced.")
    parser.add_argument("--scheduler_reduce_patience", type=int, default=5, help="Only active when scheduler is ReduceLROnPlateau. Max epoch of no improvements before LR is reduced.")
    
    # Training args.
    parser.add_argument("--batch_size", type=int, default=10, help='Batch size for the dataloader to be used in the training run.')
    parser.add_argument("--max_epoch", type=int, default=50, help='Maximum number of epochs for the training to run.')
    parser.add_argument("--eval_metric", type=str, default="f1", help='Evaluation metric to determine the best epoch and model performance.')
    parser.add_argument("--early_stop", action="store_true", help="Activates early stopping mechanism.")
    parser.add_argument("--no_early_stop", dest="early_stop", action="store_false", help="No early stopping. Train until max epoch reached.")
    parser.add_argument("--stop_patience", type=int, default=10, help='Number of epochs of no improvement in evaluation metric to trigger early stopping mechanism.')
    parser.add_argument("--stop_min_delta", type=float, default=0.0, help='The magnitude of change in evaluation metric to be considered as an improvement.')
    parser.add_argument("--stop_min_epoch", type=int, default=25, help='Number of epochs to run before early stopping logic is enabled amd patience counter starts to increase.')
    parser.add_argument("--num_workers", type=int, default=6, help='Number of PyTorch DataLoader workers.')
    parser.add_argument("--random_state", type=int, default=42, help='Number to initialise random seed.')

    # Utility args.
    parser.add_argument("--save_checkpoints", action="store_true", help="Activates checkpointing mechanism.")
    parser.add_argument("--no_save_checkpoints", dest="save_checkpoints", action="store_false", help="Deactivates checkpointing mechanism.")
    parser.add_argument("--save_bestmodel", action="store_true", help="Saves the best trained model.")
    parser.add_argument("--no_save_bestmodel", dest="save_bestmodel", action="store_false", help="Does not save the best trained model.")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help='Epoch interval at which to save checkpoints.')
    parser.add_argument("--cleanup_interval", type=int, default=0, help='Epoch interval at which to trigger memory cleanup.')
    parser.add_argument("--verbose_flag", action="store_true")
    parser.add_argument("--no_verbose_flag", dest="verbose_flag", action="store_false")
    
    parser.set_defaults(freeze_backbone=True, early_stop=True, save_checkpoints=False, save_bestmodel=False, verbose_flag=False)
    args = parser.parse_args()


    # Identify training type.
    train_stage = "head" if args.freeze_backbone else "full"
    print(f"\nTraining stage: {train_stage}.")

    # Display hyperparameters.
    print(f'\n----- Hyperparameters ----------\n')
    print(f'Optimizer: {args.optimizer_name}')
    print(f'Learning rate: {args.opt_learning_rate}')
    print(f'Weight decay: {args.opt_weight_decay}')
    print(f'Scheduler: {args.scheduler_name}')
    print(f'Layer initialisation method: {args.layer_init}')
    print(f'Batch size: {args.batch_size}')
    print(f'\n---------------\n')

    # Identify scipt location and project directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Set global seeds.
    ecg_utils.set_global_seeds(seed_num=args.random_state)

    # Load data.
    df = pd.read_csv(args.csv_path)
    labels_list = df['label'].tolist()

    # Implement k-fold cross-val if fold info are provided.
    fold_num = args.fold
    total_folds = args.total_folds
    if (fold_num is not None) and (total_folds is not None) and (total_folds != 1):

        # Generate model config + fold label.
        model_label = f'{args.config_label}_fold{fold_num}'

        # Create kfold object and extract relevant indices.
        skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=args.random_state)
        skfSplits = list(skf.split(np.zeros(len(df)), labels_list))
        train_idx, val_idx = skfSplits[fold_num]
    
    # If k-fold not provided, function with generate train-val split internally.
    else:
        train_idx = None
        val_idx = None
        model_label = f'{args.config_label}'
    
    # Setup base experiment directory.
    base_experiment_dir = os.path.join(project_dir, "experiments", f"{args.submit_timestamp}_{args.model_arch}_{train_stage}_{args.dataset_ver}")

    # Setup sub-folders
    model_cache_directory = os.path.join(base_experiment_dir, "final-models", "cache")
    model_best_directory = os.path.join(base_experiment_dir, "final-models", "best")
    log_out_directory = os.path.join(base_experiment_dir, "local-logs", "out")
    log_err_directory = os.path.join(base_experiment_dir, "local-logs", "err")
    grid_result_directory = os.path.join(base_experiment_dir, "grid-results")
    final_result_directory = os.path.join(base_experiment_dir, "final-results")
    
    # Ensures directories exist.
    os.makedirs(model_cache_directory, exist_ok=True)
    os.makedirs(model_best_directory, exist_ok=True)
    os.makedirs(log_out_directory, exist_ok=True)
    os.makedirs(log_err_directory, exist_ok=True)
    os.makedirs(grid_result_directory, exist_ok=True)
    os.makedirs(final_result_directory, exist_ok=True)

    # Identify state_dict_path if folder provided.
    if args.state_dict_folder is not None:
        state_dict_path = os.path.join(project_dir, "experiments", args.state_dict_folder, "final-models", "best", "best_model.pth")
    else:
        state_dict_path = None

    # Run training loop.
    model, bestMetricsDict, metricsHist = train_cnn_model(
        args.lmdb_path,
        args.csv_path,
        args.model_arch,
        args.optimizer_name,
        model_cache_directory,
        model_best_directory,
        batchSize=args.batch_size,
        numWorkers=args.num_workers,
        randomState=args.random_state,
        trainIdx=train_idx,
        valIdx=val_idx,
        valSplitRatio=args.val_ratio,
        freezeBackbone=args.freeze_backbone,
        initMethod=args.layer_init,
        stateDictPath=state_dict_path,
        learningRate=args.opt_learning_rate,
        weightDecay=args.opt_weight_decay,
        schedulerName=args.scheduler_name,
        schedulerStepSize=args.scheduler_step,
        schedulerGamma=args.scheduler_gamma,
        schedulerReduceFactor=args.scheduler_reduce_factor,
        schedulerReducePatience=args.scheduler_reduce_patience,
        evalMetric=args.eval_metric,
        evalPatience=args.stop_patience,
        evalMinDelta=args.stop_min_delta,
        minEpochStop=args.stop_min_epoch,
        modelLabel=model_label,
        cacheInterval=args.checkpoint_interval,
        verbose=args.verbose_flag,
        numEpochs=args.max_epoch,
        saveCheckpoint=args.save_checkpoints,
        saveBestModel=args.save_bestmodel,
        earlyStop=args.early_stop,
        cleanupInterval=args.cleanup_interval
    )

    # If implementing k-fold, cross-val, save fold-level metrics.
    if (fold_num is not None) and (total_folds is not None) and (total_folds != 1):

        # Add fold metadata to results.
        result_dict = {
            'stage': train_stage,
            'architecture': args.model_arch,
            'model_num': args.config_label,
            'fold_num': fold_num,
            'loss': bestMetricsDict['loss'],
            'accuracy': bestMetricsDict['accuracy'],
            'f1': bestMetricsDict['f1'],
            'precision': bestMetricsDict['precision'],
            'recall': bestMetricsDict['recall'],
            'best_epoch': bestMetricsDict['epoch'],
            'optimizer': args.optimizer_name,
            'learning_rate': args.opt_learning_rate,
            'weight_decay': args.opt_weight_decay,
            'scheduler': args.scheduler_name,
            'batch_size': args.batch_size,
            'init_method': args.layer_init,
            'scheduler_step_size': args.scheduler_step,
            'scheduler_gamma':args.scheduler_gamma,
            'earlystop_patience': args.stop_patience,
            'earlystop_min_epoch': args.stop_min_epoch,
            'time_total_training': bestMetricsDict['timeTotalTraining'],
            'time_epoch_average': bestMetricsDict['timeAveragePerEpoch']
        }

        # Save results.
        output_filename = f'{model_label}_metrics.json'
        filepath = os.path.join(grid_result_directory, output_filename)
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=4)
        
        print(f'\nTraining evaluation metrics for best epoch is saved at {filepath}')
    
    # Else, save metrics and training history.
    else:
        # Create dictionary to store info.
        result_dict = {
            'best_metrics': bestMetricsDict,
            'best_train_history': metricsHist
        }

        # Save results.
        output_filename = f'{model_label}_results.json'
        filepath = os.path.join(final_result_directory, output_filename)
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=4)
        
        print(f'\nTraining history and evaluation metrics for best epoch are saved at: {filepath}')
    
    if args.save_bestmodel:
        print(f'\nState dict of best epoch is saved at {model_best_directory}')
    
    print("\n")
# --- Deep learning packages ---
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna

# --- Utility packages ---
import time
import datetime
import math

# --- Helper functions ---
from .ecg_general_utils import check_runtime_device
from .ecg_dataloader import CustomSplitDataloader
from .ecg_model_utils import get_cnn_model, make_optimizer_fn, make_scheduler_fn
from .ecg_train_utils import MetricMonitor, CheckpointManager, CleanupManager

# -- References ---
# This script is a modified and more detailed version of the training script used in the INM460: Computer Vision assignment.
# The general structure is the same - two different loops for training and validation runs.
# However, substantial elements have been modified to suit this project, including making it more modular.


def train_cv_model(
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
        cleanupInterval=0,
        optuna_trial=None
):
    """
    Description.
    """
    # ------------- #
    # Preliminaries #
    # ------------- #
    # Identify runtime mode.
    device = check_runtime_device()

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
    loader_builder = CustomSplitDataloader(
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
    model = get_cnn_model(
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
    config_optimizer_fn = make_optimizer_fn(
        optimizerName,
        learningRate,
        weightDecay,
        momentum=momentum
    )
    model_optimizer = config_optimizer_fn(model)

    # Scheduler.
    config_scheduler_fn = make_scheduler_fn(
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
    modelMetricMonitor = MetricMonitor(
        metricType=evalMetric,
        metricMode=metricModeDict[evalMetric],
        patience=evalPatience,
        minDelta=evalMinDelta,
        minEpoch=minEpochStop
    )

    # Define checkpoint to enhance fault tolerance during training.
    modelCheckpoint = CheckpointManager(
        model,
        cacheDir,
        bestDir,
        modelLabel=modelLabel,
        saveInterval=cacheInterval
    )
    
    # Instantiate memory cleanup helper.
    memoryCleanupHelper = CleanupManager(verbose=verbose)

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

                if optuna_trial:
                    optuna_trial.report(epochMetricsDict[evalMetric], epoch)
                    if optuna_trial.should_prune():
                        print(f"[UPDATE!] Optuna trial prune triggered at epoch {epoch}.\n")
                        raise optuna.exceptions.TrialPruned()

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
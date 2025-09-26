# --- Baseline packages ---
import pandas as pd

# --- Deep Learning packages ---
import torch

# --- Utility packages ---
import os
import datetime
import copy
import gc



class MetricMonitor:
    def __init__(self, metricType='f1', metricMode='max', patience=5, minDelta=0.0, minEpoch=0):
        """
        Instantiates a class that monitors the a single evaluation metric to track the best epoch and
        facilitate implementation of early stopping mechanism. When called, the class takes in the epoch
        number and a dictionary of evaluation metrics as inputs. The class stores the inputs if the evaluation
        metric is the historical best.

        Arguments:
            metricType (str, optional): Evaluation metric to monitor. Currently supports loss, accuracy, f1, precision, and recall. (Default: 'f1')
            metricMode (str, optional): Whether to monitor the maximum (max) or minimum (min) of the evaluation metric. (Default: 'max')
            patience (int, optional): Number of epochs of no improvement in evaluation metric to trigger early stopping mechanism. (Default: 5)
            minDelta (float, optional): The magnitude of change in evaluation metric to be considered as an improvement. (Default: 0.0)
            minEpoch (int, optional): Number of epochs to run before early stopping logic is enabled amd patience counter starts to increase. (Default: 0)
        """
        assert metricMode in ['min', 'max'], "metricMode must be 'min' or 'max'."
        assert metricType in ['loss', 'accuracy', 'precision', 'recall', 'f1'], f"{metricType} is not supported. Try a different metric."

        self.metricType = metricType
        self.metricMode = metricMode
        self.patience = patience
        self.minDelta = minDelta
        self.minEpoch = minEpoch

        self.bestMetricVal = None
        self.bestMetricDict = None
        self.bestEpoch = 0
        self.patienceCounter = 0
        self.earlyStop = False
    
    def _check_improvement(self, currentVal):
        if self.metricMode == 'min':
            return currentVal < self.bestMetricVal - self.minDelta
        else:
            return currentVal > self.bestMetricVal + self.minDelta
    
    def __call__(self, epochNum, currentMetricDict):
        """
        Monitors the evaluation metric for the current epoch. Updates the class state to store
        the current epoch and evaluation metric dictionary if the evaluation metric is the
        historical best. Otherwise, the patience counter is increased by one if the minEpoch
        has passed.

        Arguments:
            epochNum (int): The current epoch number.
            currentMetricDict (dict): A dictionary containing evaluation metrics for the current epoch.
        """
        # Get the relevant metric value for current epoch.
        monitoredVal = currentMetricDict[self.metricType]

        # For first epoch, initialise best value and metrics dictionary.
        if self.bestMetricVal is None:
            self.bestMetricVal = monitoredVal
            self.bestMetricDict = currentMetricDict.copy()
            self.bestEpoch = epochNum
        
        # Otherwise, check whether the metric has improved.
        else:
            if self._check_improvement(monitoredVal):
                # New best: update value, dictionary and counter.
                self.bestMetricVal = monitoredVal
                self.bestMetricDict = currentMetricDict.copy()
                self.bestEpoch = epochNum
                self.patienceCounter = 0
            else:
                # No improvement: increase patience counter if training has exceeded minimum epoch.
                if epochNum >= self.minEpoch:
                    self.patienceCounter += 1

                # Update early stop bool if exceed patience limit after minimum epoch.
                if self.patienceCounter >= self.patience:
                    self.earlyStop = True
    
    def get_best_epoch(self):
        return self.bestEpoch
    
    def get_best_metrics_dict(self):
        return self.bestMetricDict



class CheckpointManager:
    def __init__(self, model, checkpointDirectory, bestDirectory, modelLabel='model', saveInterval=20):
        """
        Instantiates a class that helps with checkpointing the model state.

        Arguments:
            model (torch.nn.Module): The PyTorch model to manage checkpoints for.
            checkpointDirectory (str): Path to the directory for periodic checkpoints.
            bestDirectory (str): Path to the directory for saving the best model throughout the training loop.
            modelLabel (str, optional): A label for the model filenames. (Default: 'model').
            saveInterval (int, optional): Epoch interval at which to save checkpoints. (Default: 20).
        """
        self.model = model
        self.modelLabel = modelLabel
        self.checkpointDirectory = checkpointDirectory
        self.bestDirectory = bestDirectory
        self.saveInterval = saveInterval
        self.bestModelParams = None

        # Ensures directory exists.
        os.makedirs(self.checkpointDirectory, exist_ok=True)
        os.makedirs(self.bestDirectory, exist_ok=True)
        print("Ensured relevant directories exists.")

    def check_and_save(self, epochNum, is_best_epoch=False, save_checkpoint=False, save_bestmodel=False):
        """
        Checks conditions and saves model checkpoints and the best model.

        Arguments:
            epochNum (int): Current epoch number.
            is_best_epoch (bool, optional): Flag to indicate if the current epoch achieved the best performance on the validation set. (Default: False).
            save_checkpoint (bool, optional): Whether to save a periodic checkpoint. (Default: False).
            save_bestmodel (bool, optional): Whether to save the best model to disk if it's the best epoch. (Default: False).
        """
        # Save checkpoint at pre-determined intervals.
        if save_checkpoint and ((epochNum % self.saveInterval) == 0):
            checkpointPath = os.path.join(self.checkpointDirectory, f'{self.modelLabel}_cache_epoch_{epochNum}.pth')
            torch.save(self.model.state_dict(), checkpointPath)
            print(f"[CHECKPOINT] Saved interval model checkpoint at epoch {epochNum} to {checkpointPath}. | Timestamp: {datetime.datetime.now()}")

        # Save best model state_dict.
        if is_best_epoch:
            state_dict_copy = copy.deepcopy(self.model.state_dict())
            self.bestModelParams = state_dict_copy
            if save_bestmodel:
                bestPath = os.path.join(self.bestDirectory, f'{self.modelLabel}.pth')
                torch.save(state_dict_copy, bestPath)
    
    def get_best_params(self):
        """Returns the state_dict of the best model recorded so far in memory."""
        return self.bestModelParams



class CleanupManager:
    def __init__(self, verbose=False):
        """
        Instantiates a class that helps with memory cleanup. Whenever called, forces garbage
        collection and clear GPU cache (both MPS and CUDA).
        """
        self.verbose = verbose
    
    def cleanup_memory(self):
        """Forces garbage collection and clear GPU cache (both MPS and CUDA)."""
        # Clean up CPU and RAM.
        gc.collect()

        # Clean up GPU.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.verbose:
            print(f"[CLEANUP] Performed memory cleanup.    Timestamp: {datetime.datetime.now()}")
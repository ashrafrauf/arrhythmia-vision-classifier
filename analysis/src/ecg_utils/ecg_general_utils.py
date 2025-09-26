import numpy as np
import os
import torch
import random



def set_global_seeds(seed_num=42):
    """
    Sets the global random seeds for base Python, NumPy, and PyTorch.
    
    Arguments:
        seed_num (int): Number to initialise random seed. (Default: 42)
    
    Returns:
        None
    """
    # Set global torch seed.
    torch.manual_seed(seed_num)
    print("Global seed for torch set.")

    # Set seed for in-built python and numpy RNG.
    np.random.seed(seed_num)
    random.seed(seed_num)
    print("Global seed for numpy and python's inbuilt RNG set.")



def check_runtime_device():
    """
    Checks if GPU acceleration is available for PyTorch.
    """
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print("Runtime type:", device)

    return device



def get_dataset_paths(
        main_data_dir,
        model_arch,
        dataset_config,
        dataset_mode
):
    """
    Gets the appropriate dataset paths. For EfficientNet-B3, 300x300 image inputs are used, which is saved in
    the img_efficientnet.lmdb dataset. For all other models, 224x224 image inputs are used, which is saved in
    the img_resnet.lmdb dataset. Note: the naming convention of the LMDB datasets were due to initial experiments
    only using ResNet-18 and EfficientNet-B3 models.

    Arguments:
        main_data_dir (str): Path to main data directory.
        model_arch (str): Model architecture used for the experiment, to select the correct dataset.
        dataset_config (str): Dataset config used for the experiment.
        dataset_mode (str): Select either train or test set.
    
    Returns:
        tuple: A tuple containing the paths to the appropriate lmdb dataset and the csv metadata.
    """
    assert dataset_mode in ['train', 'test', 'full'], f"Choose between train, test or full."

    if model_arch == 'efficientnetb3':
        lmdb_path = os.path.join(main_data_dir, "processed-data", dataset_config, dataset_mode, "img_efficientnet.lmdb")
    else:
        lmdb_path = os.path.join(main_data_dir, "processed-data", dataset_config, dataset_mode, "img_resnet.lmdb")

    csv_path = os.path.join(main_data_dir, "processed-data", dataset_config, dataset_mode, "labels_keys.csv")

    return lmdb_path, csv_path
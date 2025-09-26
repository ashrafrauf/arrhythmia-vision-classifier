# --- Baseline packages ---
import pandas as pd
import numpy as np

# --- Deep Learning packages ---
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- Computer vision packages ---
from PIL import Image

# --- Utility packages ---
import re
import lmdb
import pickle
import random



class CustomImageDataset(Dataset):
    def __init__(self, rootPath, imgNames, labelDataFrame, labelCol, transform=None):
        """
        A custom PyTorch dataset that extracts the images from rootPath, using the imgNames list
        for indexxing and labelDataFrame to obtain the associated labels. Assumes the filenames
        in imgNames have the convention {imageID}_v{layoutversion}.
        
        Arguments:
            rootPath (str): Path to directory containing the images.
            imgNames (list): List of image filenames, including the file extension.
            labelDataFrame (pd.DataFrame): A DataFrame with a FileName column containing the unique ID of the images and their labels.
            labelCol (str): The name of the label column in labelDataFrame.
            transform (torchvision.transforms, optional): Transformation to be applied to the images.
        """
        self.rootPath = rootPath
        self.imgNames = imgNames
        self.transform = transform
        self.imgLabels = labelDataFrame[labelCol]

        # Make filename-label map.
        self.labelMap = dict(zip(labelDataFrame['FileName'], labelDataFrame[labelCol]))
    
    def __len__(self):
        # Get dataset length.
        return len(self.imgNames)
    
    def __getitem__(self, idx):
        # Get image.
        imagePath = self.rootPath + self.imgNames[idx]
        image = Image.open(imagePath).convert("RGB")

        # Get corresponding label.
        originalName = re.sub(r'_v\d+.*', '', self.imgNames[idx])
        if originalName in self.labelMap:
            label = self.labelMap[originalName]
        else:
            raise ValueError(f"Label for file {originalName} not found in DataFrame.")
        
        # Apply transformation, if any.
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label) # Converts label to tensor.



class CustomLMDBDataset(Dataset):
    def __init__(self, lmdbPath, csvPath, indices=None):
        """
        A custom PyTorch Dataset class that extracts data from an LMDB database by
        using an external CSV for indexing.
        Assumes:
            - Each LMDB entry is a pickled dict with keys: 'img_tensor' and 'img_label'.
            - Metadata CSV has at least a 'key' column.
        
        Arguments:
            lmdbPath (str): Path to LMDB dataset.
            csvPath (str): Path to CSV file with 'key' column.
            indices (list, optional): Subset indices used with KFold cross validation.
        """
        self.lmdbPath = lmdbPath
        self.env = None

        # Read metadata dataframe.
        df = pd.read_csv(csvPath)

        # Adjust dataframe if indices is provided.
        if indices is not None:
            df = df.loc[indices].reset_index(drop=True)
        
        # Stores a list of keys and labels.
        self.keys = df['key'].tolist()
        self.labels = df['label'].tolist()

        # Check data is loaded correctly.
        assert len(self.keys) == len(self.labels), "Mismatch between keys and labels"
    
    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.keys)
    
    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdbPath, readonly=True, lock=False)
    
    def __getitem__(self, idx):
        # Initialise in workers.
        self._init_env()

        # Get key.
        key = f"{self.keys[idx]:08}".encode('ascii')  # Keys in lmdb are ascii encoded.

        # Get image tensor and label.
        with self.env.begin() as txn:
            lmdb_value = txn.get(key)
            if lmdb_value is None:
                raise KeyError(f"Key {key.decode()} not found in LMDB.")
            
            lmdb_obj = pickle.loads(lmdb_value)
            tensor = lmdb_obj['img_tensor']
            label = lmdb_obj['img_label']

        return tensor, label
    
    def get_num_classes(self):
        """Returns the number of unique classes."""
        unique_classes = np.unique(self.labels)
        return len(unique_classes)
    
    def get_one_image(self, idx):
        """Returns a tuple containing image tensor, image label, image name for the given index."""
        # Initialise in workers.
        self._init_env()

        # Get key.
        key = f"{self.keys[idx]:08}".encode('ascii')  # Keys in lmdb are ascii encoded.

        # Get image tensor.
        with self.env.begin() as txn:
            lmdb_value = txn.get(key)
            if lmdb_value is None:
                raise KeyError(f"Key {key.decode()} not found in LMDB.")
            
            value_dict = pickle.loads(lmdb_value)
            img_tensor = value_dict['img_tensor']
            img_label = value_dict['img_label']
            img_name = value_dict['filename']

        return img_tensor, img_label, img_name



# For reproducibility. Source: https://docs.pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    


class CustomSplitDataloader:
    def __init__(
            self,
            datasetLMDBPath,
            metadataCSVPath,
            batchSize,
            numWorkers,
            randomState=42,
    ):
        """
        Creates a class to handle splitting dataset into train and validations sets during training runs.

        Arguments:
            datasetLMDBPath (str): Path to LMDB dataset.
            metadataCSVPath (str): Path to CSV file with 'key' column.
            batchSize (int): Size of each batch of inputs.
            numWorkers (int): Number of DataLoader workers to use.
            randomState (int, optional): Random seed intialisation. (Default: 42)
        """
        self.lmdbPath = datasetLMDBPath
        self.csvPath = metadataCSVPath
        self.batchSize = batchSize
        self.numWorkers = numWorkers
        self.randomState = randomState
    
    def build_loaders(self, trainIdx=None, valIdx=None, testSetSize=0.2, returnSplitDataset=False, returnSplitSize=False):
        """
        Builds train and validation DataLoaders.

        Arguments:
            trainIdx (list, optional): Indices for the training split. Usually used in conjunction with kfold cross-validation. (Default: None)
            valIdx (list, optional): Indices for the validation split. Usually used in conjunction with kfold cross-validation.(Default: None)
            testSetSize (float, optiona): Size of validation set. Ignored if train and val indices are passed. (Default:0.2)
            returnSplitDataset (bool, optional): If True, return the split datasets. (Default: False)
            returnSplitSize (bool, optional): If True, return number of samples in each split. (Default: False)

        Returns:
            SplitDataLoader (dict): {'train': DataLoader, 'val': DataLoader}.
            SplitDataset (dict, optional): {'train': Dataset, 'val': Dataset}.
            SplitDatasetSize (dict, optional): {'train': int, 'val': int}.
        """
        # If training and validation indexes are not provided, create stratified split of data.
        if (trainIdx is None) or (valIdx is None):
            df = pd.read_csv(self.csvPath)
            labelList = df['label'].tolist()

            trainIdx, valIdx = train_test_split(
                range(len(labelList)),
                test_size=testSetSize,
                stratify=labelList,
                shuffle=True,
                random_state=self.randomState
            )
            print("Internally generated stratified dataset for training and validation splits.\n")
        
        # Otherwise, use provided training and validation indices.
        else:
            print("Used provided indices for training and validation splits.\n")
        
        # Split dataset based on generated or provided indices.
        splitDataset = {
            'train': CustomLMDBDataset(self.lmdbPath, self.csvPath, indices=trainIdx),
            'val': CustomLMDBDataset(self.lmdbPath, self.csvPath, indices=valIdx),
        }

        # Create torch random generator instance to ensure consistent data shuffling.
        shuffle_gen = torch.Generator()
        shuffle_gen.manual_seed(self.randomState)

        # Identify common dataloader arguments.
        splitloader_kwargs = {
            'batch_size': self.batchSize,
            'num_workers': self.numWorkers,
            'persistent_workers': self.numWorkers > 0,
        }
        
        # Conditionally add worker-related arguments if num_workers > 0
        if self.numWorkers > 0:
            splitloader_kwargs['worker_init_fn'] = seed_worker
            splitloader_kwargs['generator'] = shuffle_gen

        # Create training and validation DataLoaders.
        splitDataLoader = {
            'train': DataLoader(splitDataset['train'], shuffle=True, **splitloader_kwargs),
            'val': DataLoader(splitDataset['val'], shuffle=False, **splitloader_kwargs)
        }
        returnOutputList = [splitDataLoader]

        # Optionally return split datasets.
        if returnSplitDataset:
            returnOutputList.append(splitDataset)

        # Optionally return number of samples in each dataset.
        if returnSplitSize:
            splitSizes = {
                'train': len(splitDataset['train']),
                'val': len(splitDataset['val'])
            }
            returnOutputList.append(splitSizes)

        return tuple(returnOutputList) if len(returnOutputList)>1 else returnOutputList[0]
    
    def get_num_classes(self):
        """Returns the number of unique classes obtained from the CSV file."""
        df = pd.read_csv(self.csvPath)
        numClasses = df['label'].nunique()
        return numClasses
# --- Baseline packages ---
import pandas as pd

# --- Deep Learning packages ---
from torchvision import transforms
from PIL import Image

# --- Utility packages ---
import os
import glob
import pickle
import datetime
import lmdb
from tqdm import tqdm



# The function below relied on the package's documentation, found here: http://www.lmdb.tech/doc/starting.html
# Also inspiration was taken from this forum post: https://stackoverflow.com/questions/39557118/creating-lmdb-file-in-the-right-way
def generate_lmdb_dataset(
        image_dir,
        metadata_filepath,
        output_dir,
        label_col='Rhythm_L1_Enc'
):
    """
    Generates LMDB datasets using PNG image files stored in image_dir. Two LMDB datasets are created:
    one storing 224x224 images (saved as img_resnet.lmdb) and another storing 300x300 images (saved as
    img_efficientnet.lmdb). Each image is stored as a PyTorch tensor together with its associated metadata
    (labels, filenames) retrieved from metadata_filepath. Only resizing and to tensor operations were applied,
    no other data augmentations were performed. (Note: the naming convention of the LMDB datasets were due to
    initial experiments only using ResNet-18 and EfficientNet-B3 models.)

    Arguments:
        image_dir (str): Path to the directory containing the raw image files. Expected image filenames follow the pattern "{filename}_v*.png"
        metadata_filepath (str): Path to the CSV file containing the list of image names under the 'FileName' column to match images and the label column.
        output_dir (str): Path to the directory where the LMDB databases will be saved. The function will create this directory if it doesn't exist.
        label_col (str, optional): The name of the column in metadata_filepath that contains the labels for the images. (Default: 'Rhythm_L1_Enc')
    
    Returns:
        list: A list of metadata information for the dataset containing the filenames, labels, and unique ID mappings.
    """
    # Ensure output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Create relevant filepaths.
    lmdb_resnet_path = os.path.join(output_dir, "img_resnet.lmdb")
    lmdb_efficientnet_path = os.path.join(output_dir, "img_efficientnet.lmdb")

    # Make a mapping for fast lookup.
    metadata_df = pd.read_csv(metadata_filepath)
    file_label_map = dict(zip(metadata_df['FileName'], metadata_df[label_col]))

    # Statistics to normalise the images. Currently using imageNet values.
    imageNet_means = [0.485, 0.456, 0.406]
    imageNet_stds = [0.229, 0.224, 0.225]

    # Identify relevant transformation - resizing and to tensor.
    data_transforms_resnet = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(imageNet_means, imageNet_stds)
    ])
    
    data_transforms_efficientnet = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(imageNet_means, imageNet_stds)
    ])


    # ---------------- #
    # Dataset Creation #
    #----------------- #
    # Identify key dataset parameters.
    map_size = 24 * 1024 ** 3   # Maximum LMDB size. To adjust if not enough.
    batch_size = 1000           # Commit interval.

    # Get total number of files (for nicer progress bar).
    all_images = []
    for unique_file in file_label_map.keys():
        pattern = os.path.join(image_dir, f"{unique_file}_v*.png")
        all_images.extend(glob.glob(pattern))
    print(f"\n[{datetime.datetime.now()}]   Packing {len(all_images)} files into LMDB datasets saved at {output_dir}")

    # Global index as LMDB key.
    idx = 0

    # Instantiate empty list to store metadata.
    img_metadata = []

    # Create LMDB environment.
    env_resnet = lmdb.open(lmdb_resnet_path, map_size=map_size)
    env_efficientnet = lmdb.open(lmdb_efficientnet_path, map_size=map_size)

    # Check if there is already data to adjust the idx using resnet database.
    with env_resnet.begin(write=False) as txn:
        # Get statistics from the database.
        stats = txn.stat()
        num_items = stats['entries']
        print(f"[{datetime.datetime.now()}]   Found {num_items} items in the existing LMDB.")
    
    # Correct for starting index.
    idx = num_items
    print(f"Starting index is: {idx}")
    
    # Start first transaction.
    txn_resnet = env_resnet.begin(write=True)
    txn_efficientnet = env_efficientnet.begin(write=True)

    for img_path in tqdm(all_images):
        # Extract filename prefix to get the label.
        basename = os.path.basename(img_path)
        unique_filename, file_prefix = basename.split('_v')
        img_label = file_label_map[unique_filename]

        # Load and transform image.
        img = Image.open(img_path).convert('RGB')
        img_tensor_resnet = data_transforms_resnet(img)
        img_tensor_efficientnet = data_transforms_efficientnet(img)

        # Store as (tensor, label).
        key = f"{idx:08}".encode('ascii')
        
        value_resnet = pickle.dumps({
            'img_tensor': img_tensor_resnet,
            'img_label': img_label,
            'filename': basename
        })
        txn_resnet.put(key, value_resnet)

        value_efficientnet = pickle.dumps({
            'img_tensor': img_tensor_efficientnet,
            'img_label': img_label,
            'filename': basename
        })
        txn_efficientnet.put(key, value_efficientnet)

        # Save metadata.
        img_metadata.append({
            'index': idx,
            'key': key.decode('ascii'),
            'label': img_label,
            'filename': unique_filename,
            'prefix': 'v' + file_prefix.replace(".png", ""),
            'versionname': basename
        })

        # Increase global index.
        idx += 1

        # Batch commit.
        if idx % batch_size == 0:
            txn_resnet.commit()
            txn_efficientnet.commit()

            txn_resnet = env_resnet.begin(write=True)
            txn_efficientnet = env_efficientnet.begin(write=True)
    
    # Commit any leftover items
    txn_resnet.commit()
    txn_efficientnet.commit()
    
    env_resnet.close()
    env_efficientnet.close()

    print(f'\n[{datetime.datetime.now()}]   Done! Packed {idx} items to {output_dir}')

    return img_metadata
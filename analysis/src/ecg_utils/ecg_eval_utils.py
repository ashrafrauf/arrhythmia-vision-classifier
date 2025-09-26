# --- Baseline packages ---
import numpy as np

# --- Statistical packages ---
from sklearn import metrics

# --- Deep learning packages ---
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- Visualisation packages ---
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

# --- Utility packages ---
from tqdm import tqdm
import datetime

# --- Helper functions ---
from .ecg_dataloader import CustomLMDBDataset



def get_accuracy_score(true_label_array, pred_label_array, label_enc):
    """
    Calculates per-class and overall accuracy scores.

    Arguments:
        true_label_array (np.array): An array of true labels (encoded as integers).
        pred_label_array (np.array): An array of predicted labels (encoded as integers).
        label_enc (LabelEncoder): A fitted LabelEncoder object used in the pre-processing step to encode the labels.

    Returns:
        dict: A dictionary containing accuracy scores. Keys are original label names for class-level scores, and 'macro_avg' for the overall accuracy.
    """
    acc_scores_dict = {}
    nunique_labels = len(np.unique(true_label_array))

    # Store class-level scores.
    for i in range(nunique_labels):
        label_name = label_enc.inverse_transform([i]).item()
        acc_score = metrics.accuracy_score(
            true_label_array[true_label_array==i],
            pred_label_array[true_label_array==i],
            normalize=True
        )
        acc_scores_dict[label_name] = acc_score
    
    # Store macro-average scores.
    acc_scores_dict['macro_avg'] = metrics.accuracy_score(true_label_array, pred_label_array)

    return acc_scores_dict



def get_precision_recall_f1_scores(true_label_array, pred_label_array, label_enc, beta=1.0):
    """
    Calculates per-class and macro-averaged precision, recall, and F1-scores per class, along with their support.

    Arguments:
        true_label_array (np.array): An array of true labels (encoded as integers).
        pred_label_array (np.array): An array of predicted labels (encoded as integers).
        label_enc (LabelEncoder): A fitted LabelEncoder object used in the pre-processing step to encode the labels.
        beta (float, optional): The beta parameter for the F-beta score. (Default: 1.0)

    Returns:
        dict: Per-class and macro-average precision scores.
        dict: Per-class and macro-average recall scores.
        dict: Per-class and macro-average F1 scores.
        dict: Number of samples per class.
    """
    # Store class-level scores.
    prec_scores, rec_scores, f1_scores, n_samples = metrics.precision_recall_fscore_support(
        true_label_array,
        pred_label_array,
        beta=beta,
        average=None,
        zero_division=0
    )
    prec_scores_dict = dict(zip(label_enc.classes_, prec_scores.tolist()))
    rec_scores_dict = dict(zip(label_enc.classes_, rec_scores.tolist()))
    f1_scores_dict = dict(zip(label_enc.classes_, f1_scores.tolist()))
    n_samples_dict = dict(zip(label_enc.classes_, n_samples.tolist()))

    # Store macro-average scores.
    prec_scores_dict['macro_avg'] = float(np.mean(prec_scores))
    rec_scores_dict['macro_avg'] = float(np.mean(rec_scores))
    f1_scores_dict['macro_avg'] = float(np.mean(f1_scores))
    n_samples_dict['total'] = float(np.sum(n_samples))

    return prec_scores_dict, rec_scores_dict, f1_scores_dict, n_samples_dict



def get_auroc_score(true_label_array, softmax_label_array, label_enc):
    """
    Calculates per-class and macro-averaged Area Under the Receiver Operating Characteristic (AUROC) score.

    Arguments:
        true_label_array (np.array): An array of true labels (encoded as integers).
        softmax_label_array (np.array): A 2D array of predicted probabilities (softmax outputs), where each row corresponds to a sample and columns
                                        correspond to class probabilities. The order of columns should match the integer encoding of classes.
        label_enc (LabelEncoder): A fitted LabelEncoder object used in the pre-processing step to encode the labels.

    Returns:
        dict: Per-class and macro-average AUROC scores.
    """
    n_classes = len(np.unique(true_label_array))
    true_labels_onehot = F.one_hot(torch.from_numpy(true_label_array), n_classes).numpy()

    # Store class-level scores.
    auc_scores = metrics.roc_auc_score(
        true_labels_onehot,
        softmax_label_array,
        average=None,
        multi_class='ovr'
    )
    auc_scores_dict = dict(zip(label_enc.classes_, auc_scores.tolist()))

    # Store macro-average scores.
    auc_scores_dict['macro_avg'] = float(np.mean(auc_scores))

    return auc_scores_dict



def get_model_predictions(dataloader, model, device='cpu'):
    """
    Extracts true labels, softmax probabilities, and predicted labels for a PyTorch model.

    Arguments:
        dataloader (torch.utils.data.DataLoader): The DataLoader containing the dataset to get predictions for.
        model (torch.nn.Module): The PyTorch model to use for inference.
        device (str, optional): Runtime type. (Default: 'cpu')

    Returns:
        np.array: An array of true labels (encoded as integers).
        np.array: A 2D array of predicted probabilities (softmax outputs), where each row corresponds to a sample and columns correspond to class probabilities.
        np.array: An array of predicted labels (encoded as integers).
    """
    true_labels = []
    softmax_labels = []
    predicted_labels = []

    model = model.to(device).eval()

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.inference_mode():
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
        
        true_labels.append(labels.cpu())
        softmax_labels.append(probs.cpu())
        predicted_labels.append(preds.cpu())
    
    true_labels = torch.cat(true_labels).numpy()
    softmax_labels = torch.cat(softmax_labels).numpy()
    predicted_labels = torch.cat(predicted_labels).numpy()

    return true_labels, softmax_labels, predicted_labels



def evaluate_model_predictions(
        lmdb_path,
        csv_path,
        model,
        label_enc,
        loader_batch_size = 64,
        device = 'cpu',
        fscore_beta = 1.0,
        metadata = False
):
    """
    Wrapper function to automate the calculation of various evaluation metrics. Currently
    supporting calculations of accuracy, precision, recall, F1, and AUROC scores. Both
    per-class and overall scores are calculated, where relevant scores are aggregated 
    using the macro-average method.

    Arguments:
        lmdbPath (str): Path to LMDB dataset.
        csvPath (str): Path to CSV file with 'key' column.
        model (torch.nn.Module): PyTorch model to be evaluated.
        label_enc (LabelEncoder): A fitted LabelEncoder object used in the pre-processing step to encode the labels.
        loader_batch_size (int, optional): Batch size for the DataLoader. (Default: 64)
        device (str, optional): Runtime type. (Default: 'cpu')
        fscore_beta (float, optional): The beta parameter for the F-beta score. (Default: 1.0)
        metadata (bool, optional): Flag to also return the total number of samples in the dataset. (Default: False)

    Returns:
        dict: Contains the outputs of the PyTorch model - true labels, softmax probabilities, and predicted labels.
        dict: Contains a nested dictionary for each evaluation metric, with per-class and macro-averaged scores.
    """
    lmdb_dataset = CustomLMDBDataset(lmdb_path, csv_path)
    lmdb_dataloader = DataLoader(lmdb_dataset, shuffle=False, batch_size=loader_batch_size)

    print(f'\n[{datetime.datetime.now()}]    Getting model predictions...')
    true_labels, softmax_labels, predicted_labels = get_model_predictions(
        lmdb_dataloader,
        model,
        device=device
    )
    model_outputs = {'true_labels': true_labels, 'pred_labels': predicted_labels, 'label_probs': softmax_labels}

    print(f'\n[{datetime.datetime.now()}]    Evaluating model predictions...')
    acc_scores_dict = get_accuracy_score(
        true_labels,
        predicted_labels,
        label_enc
    )

    prec_scores_dict, rec_scores_dict, f1_scores_dict, n_samples_dict = get_precision_recall_f1_scores(
        true_labels,
        predicted_labels,
        label_enc,
        beta=fscore_beta
    )

    auc_scores_dict = get_auroc_score(
        true_labels,
        softmax_labels,
        label_enc
    )

    metric_scores_dict = {
        'accuracy': acc_scores_dict,
        'precision': prec_scores_dict,
        'recall': rec_scores_dict,
        'f1': f1_scores_dict,
        'auroc': auc_scores_dict,
        'support': n_samples_dict
    }

    if metadata:
        metric_scores_dict['n_samples'] = len(lmdb_dataset)

    print(f'\n[{datetime.datetime.now()}]    Evaluation summary:')
    print(metrics.classification_report(true_labels, predicted_labels, target_names=label_enc.classes_))

    return model_outputs, metric_scores_dict



def plot_roc_auprc(true_label_array, softmax_label_array, label_enc, viz_title=None, adj_title_space=0.85):
    """
    Plots the AUROC and AUPRC curves side-by-side in a figure with two subplots using matplotlib.

    Arguments:
        true_label_array (np.array): An array of true labels (encoded as integers).
        softmax_label_array (np.array): A 2D array of predicted probabilities (softmax outputs), where each row corresponds to a sample and columns
                                        correspond to class probabilities. The order of columns should match the integer encoding of classes.
        label_enc (LabelEncoder): A fitted LabelEncoder object used in the pre-processing step to encode the labels.
        viz_title (str, optional): Title of the displayed figured. (Default: None)
        adj_title_space (float, optional): Spacing between title and subplots. (Default: 0.8)
    
    Returns:
        None: The figure is immediately displayed.
    """
    n_classes = len(np.unique(true_label_array))
    true_labels_onehot = F.one_hot(torch.from_numpy(true_label_array), n_classes).numpy()

    fig, axs = plt.subplots(1,2, figsize=(12,5))

    for i in range(n_classes):
        label_name = label_enc.inverse_transform([i]).item()
        fpr, tpr, _ = metrics.roc_curve(true_labels_onehot[:, i], softmax_label_array[:, i])
        roc_auc = metrics.auc(fpr, tpr)
        axs[0].plot(fpr, tpr, label=f"{label_name} (AUROC: {roc_auc:.3f})")

        precision, recall, _ = metrics.precision_recall_curve(true_labels_onehot[:, i], softmax_label_array[:, i])
        auprc = metrics.average_precision_score(true_labels_onehot[:, i], softmax_label_array[:, i])
        axs[1].plot(recall, precision, label=f'{label_name} (AUPRC: {auprc:.3f})')
    
    axs[0].legend(frameon=False)
    axs[0].set_title("Area under the Receiver Operating Characteristic Curve")

    axs[1].legend(frameon=False)
    axs[1].set_title("Area under the Precision-Recall Curve")

    if viz_title is not None:
        fig.suptitle(f'{viz_title}', fontsize=14, fontweight='heavy')
        fig.subplots_adjust(top=adj_title_space)
    
    plt.show()

    return None
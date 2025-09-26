# --- Baseline packages ---
import numpy as np

# --- Deep learning packages ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
import cv2
from PIL import Image

# --- Visualisation packages ---
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

# --- Utility packages ---
import copy
import os

# --- Helper functions ---
from .ecg_dataloader import CustomLMDBDataset
from .ecg_general_utils import get_dataset_paths


# --- References ---
# Code within this script was adapted from various libraries and implementations.
# The main reference for Grad-CAM was the pytorch_grad_cam repo, available here: https://github.com/jacobgil/pytorch-grad-cam
# The main reference for Guided Backpropagation was from the pytorch-cnn-visualisation repo, available here: https://github.com/utkuozbulak/pytorch-cnn-visualizations/tree/master
# This tutorial was also helpful when getting started: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82


class CustomGradCam:
    def __init__(self, model, target_layer, device='cpu'):
        """
        Instantiates a class to implement Grad-CAM for a PyTorch model. Hooks are registered when the
        generate_map function is called and is removed at the end of the function.

        Arguments:
            model (torch.nn.Module): PyTorch model to implement the Grad-CAM method.
            target_layer (torch.nn.Module): The layer to attach hooks and extract the gradient and activation maps.
            device (str): Runtime environment.
        """
        self.model = model.to(device).eval()
        self.target_layer = target_layer
        self.device = device

        self.feature_map = None
        self.gradient = None
    
    def _save_activation_and_gradient(self, module, input, output):
        """Saves the feature map and registers a hook to collect gradients."""
        # Due to https://github.com/pytorch/pytorch/issues/61519, avoid using register_full_backward_hooks function.
        
        # Collect feature map of target layer.
        self.feature_map = output
        
        # Collect output gradients of target layer.
        if output.requires_grad:
            def _store_grad(grad):
                self.gradient = grad
            output.register_hook(_store_grad)
        else:
            print("Warning: Target layer output does not require grad. Grad-CAM might not work.")
    
    def _remove_hooks(self):
        self.forward_handle.remove()
    
    def generate_map(self, input_tensor, target_class=None, skew=False):
        """
        Generates a saliency map by attaching hooks to extract activation map and gradients
        at the appropriate layer. Then takes the average gradient for each channel and multiplies
        with the activations, and then averages the activations across channels. Subsequently,
        remove hooks to avoid unintentional buildup. The values are normalised at the end, before
        skew correction is applied (if any).

        Arguments:
            input_tensor (tensor): Image to be used for generating saliency map.
            target_class (int, optional): Target class for backpropagation. If None, the predicted class is used. (Default: None)
            skew (bool, optional): Flag to correct for skewness in activations by taking the square root. (Default: False)
        
        Returns:
            np.array: An array containing the Grad-CAM values.
        """
        # Register hook. Due to https://github.com/pytorch/pytorch/issues/61519, avoid using register_full_backward_hooks function.
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activation_and_gradient)

        try:
            # Ensure input tensor shape suitable for forward pass.
            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Clear previous values.
            self.model.zero_grad()
            self.feature_map = None
            self.gradient = None
            self.target_class = None

            # Forward pass. Assumes a batch size of 1.
            output_tensor = self.model(input_tensor)
            if target_class is None:
                target_class = output_tensor.argmax(dim=1).item()
            self.target_class = target_class
            
            # Backward pass. Calculates the gradients of the selected logit wrt differentiable parameters.
            output_tensor[0, target_class].backward()

            # Collect gradients and feature map. Use detach and clone to avoid bugs.
            grad = self.gradient.clone().detach().cpu()  # [1, C, H, W]
            fmap = self.feature_map.clone().detach().cpu()  # [1, C, H, W]

            # Get average gradients for each feature map.
            # > I.e., average across height and width dimensions.
            weights = grad.mean(dim=(2,3,), keepdim=True)  # [1, C, 1, 1]

            # For each feature map (i.e. channel), multiple with its weights and sum all values.
            cam = (fmap * weights).sum(dim=1, keepdim=True)  # [1, 1, H, W]

            # Get positive contributions.
            cam = F.relu(cam)

            # Remove batch and channel dimensions. Convert to numpy.
            cam = cam.squeeze().numpy()

            # Normalize the heatmap to be between 0 and 1.
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Add epsilon to prevent division by zero

            if skew:
                cam = np.sqrt(cam)

            return cam
        
        finally:
            # Ensure hook is removed even if an error occurs.
            self._remove_hooks()



class GuidedReLUFunction(torch.autograd.Function):
    """
    An activation function that acts as ReLU in the forward pass and suppresses
    negative gradients in the backward pass.
    """
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        output_tensor = torch.clamp(input_tensor, min=0.0)
        return output_tensor
    
    @staticmethod
    def backward(ctx, grad_out):
        input_tensor, = ctx.saved_tensors
        grad_mask = (input_tensor > 0) & (grad_out > 0)
        grad_in = grad_out * grad_mask
        return grad_in



class CustomGuidedReLUModule(nn.Module):
    """
    Module wrapper for the Guided ReLU activation function, to be used to
    replace exisiting activation functions within the PyTorch models.
    """
    def __init__(self):
        super(CustomGuidedReLUModule, self).__init__()

    def forward(self, input_img):
        return GuidedReLUFunction.apply(input_img)



class CustomGuidedBackprop:
    def __init__(self, model, device='cpu'):
        """
        Instantiates a class to implement the Guided Backpropagation method for a PyTorch
        model. Currently only supports MobileNetV3, ResNet, EfficientNet, and ConvNeXt
        model families. Takes a simplistic approach by replacing all activations with a
        Guided ReLU, potentially distorting the gradient propagations and decoupling the
        saliency map generated from the decision-making process.

        Arguments:
            model (torch.nn.Module): PyTorch model to implement the Guided Backpropagation method.
            device (str): Runtime environment.
        """
        model_copy = copy.deepcopy(model)
        self.model = model_copy.to(device).eval()
        self.device = device
    
    def _implement_guided_activation_recursive(self, model, act_ori, act_guided):
        """Replaces original activation with Guided ReLU."""
        for name, layer in model.named_children():
            if isinstance(layer, act_ori):
                setattr(model, name, act_guided())
            self._implement_guided_activation_recursive(layer, act_ori, act_guided)
    
    def _restore_activation_recursive(self, model, act_ori, act_guided):
        """Restores original activation function."""
        for name, layer in model.named_children():
            if isinstance(layer, act_guided):
                if isinstance(model, models.ResNet):
                    setattr(model, name, act_ori(inplace=True))
                if isinstance(self.model, models.ConvNeXt):
                    setattr(model, name, act_ori())
            self._restore_activation_recursive(layer, act_ori, act_guided)
    
    def _get_model_activation(self):
        """Defines the activation functions to be replaced for each architecture family."""
        if isinstance(self.model, models.ResNet):
            act_ori = nn.ReLU
            act_guided = CustomGuidedReLUModule
        if isinstance(self.model, models.ConvNeXt):
            act_ori = nn.GELU
            act_guided = CustomGuidedReLUModule
        if isinstance(self.model, models.EfficientNet):
            act_ori = nn.SiLU
            act_guided = CustomGuidedReLUModule
        if isinstance(self.model, models.MobileNetV3):
            act_ori = nn.Hardswish
            act_guided = CustomGuidedReLUModule
        
        return act_ori, act_guided
    
    def generate_map(self, input_tensor, target_class=None):
        """
        Generates a saliency map by replacing all activation functions with the Guided ReLU.
        Then backpropagates the gradient to the input tensor. Finally, restores the
        original activation functions. The values are normalised at the end.

        Arguments:
            input_tensor (tensor): Image to be used for generating saliency map.
            target_class (int, optional): Target class for backpropagation. If None, the predicted class is used. (Default: None)
        
        Returns:
            np.array: An array containing the Guided Backpropagation values.

        """
        original_activation, guided_activation = self._get_model_activation()
        
        self._implement_guided_activation_recursive(self.model, original_activation, guided_activation)

        if len(input_tensor.shape)==3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        self.model.zero_grad()

        output_tensor = self.model(input_tensor)
        if target_class is None:
            target_class = output_tensor.argmax(dim=1).item()
        self.target_class = target_class
        output_tensor[0, target_class].backward()

        input_tensor_grads = input_tensor.grad.detach().clone().cpu()
        input_tensor_grads = torch.clamp(input_tensor_grads, min=0.0)  # [1, C, H, W]

        cam = input_tensor_grads.squeeze().numpy().transpose(1, 2, 0)  # [H, W, C]

        cam = np.mean(cam, axis=2)  # [H, W]

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        self._restore_activation_recursive(self.model, original_activation, guided_activation)

        return cam



class CustomGuidedGradCam:
    def __init__(self, model, target_layer, device='cpu'):
        """
        Instantiates a class to implement the Guided Grad-CAM method for a PyTorch
        model. Currently only supports MobileNetV3, ResNet, EfficientNet, and ConvNeXt
        model families. Takes a simplistic approach by replacing all activations with a
        Guided ReLU, potentially distorting the gradient propagations and decoupling the
        saliency map generated from the decision-making process.

        Arguments:
            model (torch.nn.Module): PyTorch model to implement the Guided Grad-CAM method.
            target_layer (torch.nn.Module): The layer to attach hooks and extract the gradient and activation maps.
            device (str): Runtime environment.
        """
        self.target_layer = target_layer
        self.device = device

        self.gradcam = CustomGradCam(model, self.target_layer, self.device)
        self.guidedbackprop = CustomGuidedBackprop(model, self.device)
    
    def generate_map(self, input_tensor, target_class=None, skew=False):
        """
        Generates a saliency map by combining Grad-CAM and Guided Backpropagation
        via a pointwise multiplication of the maps generated by both methods. The 
        Grad-CAM resolution is rescaled to the input resolution using OpenCV's
        resize function. The values are normalised at the end, before skew correction
        is applied (if any).

        Arguments:
            input_tensor (tensor): Image to be used for generating saliency map.
            target_class (int, optional): Target class for backpropagation. If None, the predicted class is used. (Default: None)
            skew (bool, optional): Flag to correct for skewness in Grad-CAM by taking the square root. (Default: False)
        
        Returns:
            np.array: An array containing the Guided Grad-CAM values.
        """
        map_gradcam = self.gradcam.generate_map(input_tensor, target_class)
        map_guidedbackprop = self.guidedbackprop.generate_map(input_tensor, target_class)

        h_img, w_img = map_guidedbackprop.shape
        map_gradcam_rescaled = cv2.resize(map_gradcam, (w_img, h_img))

        map_combined = map_gradcam_rescaled * map_guidedbackprop
        map_combined = (map_combined - map_combined.min()) / (map_combined.max() - map_combined.min() + 1e-8)

        if skew:
            map_combined = np.sqrt(map_combined)

        return map_combined



def show_cam_on_image(input_tensor, cam, alpha=0.4, adj_bright=0.0):
    """
    Overlays the saliency map generated onto the input image. The saliency map
    is resized using OpenCV's resize function and applied OpenCV's colormap_jet
    scheme to highlight the activated regions. The input image and saliency map
    is then combined using OpenCV's addWeighted function.

    Arguments:
        input_tensor (tensor): Image to be used for generating saliency map.
        cam (np.array): Array containing the saliency values.
        alpha (float, optional): Controls the relative opacity of the input and saliency map. (Default: 0.4)
        adj_bright (float, optional): Contols the brightness of the image. (Default: 0.0)
    
    Returns:
        np.array: Containes the values representing the combined image with the saliency map overlay. The values are in the range [0, 255] and have a data type of np.uint8.
    """
    if len(input_tensor.shape)==4:
        input_tensor = input_tensor.squeeze(0)
    input_np = input_tensor.cpu().numpy().transpose(1,2,0)

    h_img, w_img, c_img = input_np.shape
    cam_rescaled = cv2.resize(cam, (w_img, h_img))

    cam_coloured = cv2.applyColorMap(np.uint8(255 * cam_rescaled), cv2.COLORMAP_JET)
    cam_coloured = cv2.cvtColor(cam_coloured, cv2.COLOR_BGR2RGB)

    overlaid_img = cv2.addWeighted(input_np, 1 - alpha, cam_coloured, alpha, adj_bright)

    return overlaid_img



def compare_cam_one_image(
        lmdb_path,
        csv_path,
        data_idx,
        model,
        target_layer,
        device='cpu',
        target_class=None,
        viz_title=None,
        adj_title_space=0.8,
        adj_viz_alpha=0.4,
        adj_viz_bright=0,
        skew=False
):
    """
    Compares the output of Grad-CAM and Guided Grad-CAM in one figure for analysis, plotted using matplotlib.

    Arguments:
        lmdbPath (str): Path to LMDB dataset.
        csvPath (str): Path to CSV file with 'key' column.
        data_idx (int): Dataset index of the input to be used.
        model (torch.nn.Module): PyTorch model to implement the Guided Grad-CAM method.
        target_layer (torch.nn.Module): The layer to attach hooks and extract the gradient and activation maps.
        device (str, optional): Runtime environment. (Default: 'cpu')
        target_class (int, optional): Target class for backpropagation. If None, the predicted class is used. (Default: None)
        viz_title (str, optional): Title of the displayed figured. (Default: None)
        adj_title_space (float, optional): Spacing between title and subplots. (Default: 0.8)
        alpha (float, optional): Controls the relative opacity of the input and saliency map. (Default: 0.4)
        adj_bright (float, optional): Contols the brightness of the image. (Default: 0.0)
        skew (bool, optional): Flag to correct for skewness in Grad-CAM by taking the square root. (Default: False)
    
    Returns:
        None: The figure is immediately displayed.
    """
    # Get relevant input.
    lmdb_dataset = CustomLMDBDataset(lmdb_path, csv_path)
    if data_idx > len(lmdb_dataset):
        raise ValueError(f'Index is out of bounds. Dataset has only {len(lmdb_dataset)} samples.')
    
    img_tensor, img_label, img_name = lmdb_dataset.get_one_image(data_idx)
    if len(img_tensor.shape)==3:
        img_tensor = img_tensor.unsqueeze(0)

    grayscale_transform = transforms.Grayscale(num_output_channels=1)
    grayscale_input = grayscale_transform(img_tensor)

    # Get model prediction.
    model = model.to(device).eval()
    input = img_tensor.to(device)
    output = model(input)
    _, pred = torch.max(output, 1)

    # Generate and visualise Grad-CAM and Guided Grad-CAM saliency maps.
    normal_gradcam = CustomGradCam(model, target_layer, device=device)
    normal_gradcam_map = normal_gradcam.generate_map(img_tensor, target_class, skew=skew)
    gradcam_img = show_cam_on_image(grayscale_input, normal_gradcam_map, alpha=adj_viz_alpha, adj_bright=adj_viz_bright)

    guided_gradcam = CustomGuidedGradCam(model, target_layer, device=device)
    guided_gradcam_map = guided_gradcam.generate_map(img_tensor, target_class, skew=skew)
    guided_gradcam_img = show_cam_on_image(grayscale_input, guided_gradcam_map, alpha=adj_viz_alpha, adj_bright=adj_viz_bright)

    # Display the maps generated.
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    if viz_title is not None:
        fig.suptitle(f"{viz_title}\n Actual: {img_label}   |    Predicted: {pred.item()}", fontsize=14, fontweight='heavy')
    else:
        fig.suptitle(f"{img_name}\n Actual: {img_label}   |    Predicted: {pred.item()}", fontsize=14, fontweight='heavy')

    axs[0].imshow(gradcam_img)
    axs[0].axis("off")
    axs[0].set_title("Normal Grad-CAM")

    axs[1].imshow(guided_gradcam_img)
    axs[1].axis("off")
    axs[1].set_title("Guided Grad-CAM")

    fig.subplots_adjust(top=adj_title_space)
    plt.show()

    return None



def generate_guidedgradcam_image(
        main_data_dir,
        model_arch,
        dataset_config,
        dataset_mode,
        data_idx,
        model,
        target_layer,
        device='cpu',
        target_class=None,
        adj_viz_alpha=0.4,
        adj_viz_bright=0,
        skew=False
):
    """
    Wrapper function to automate the pipeline of generating Guided Grad-CAM images for analysis. 

    Arguments:
        main_data_dir (str): Path to main data directory.
        model_arch (str): Model architecture used for the experiment, to select the correct dataset.
        dataset_config (str): Dataset config used for the experiment.
        dataset_mode (str): Select either train or test set.
        data_idx (int): Dataset index of the input to be used.
        model (torch.nn.Module): PyTorch model to implement the Guided Grad-CAM method.
        target_layer (torch.nn.Module): The layer to attach hooks and extract the gradient and activation maps.
        device (str, optional): Runtime environment. (Default: 'cpu')
        target_class (int, optional): Target class for backpropagation. If None, the predicted class is used. (Default: None)
        alpha (float, optional): Controls the relative opacity of the input and saliency map. (Default: 0.4)
        adj_bright (float, optional): Contols the brightness of the image. (Default: 0.0)
        skew (bool, optional): Flag to correct for skewness in Grad-CAM by taking the square root. (Default: False)
    
    Returns:
         np.array: Containes the values representing the combined image with the saliency map overlay. The values are in the range [0, 255] and have a data type of np.uint8.
         int: Actual label of the input.
         int: Predicted label of the input.
         str: Input ID.
    """
    # Get relevant input.
    lmdb_path, csv_path = get_dataset_paths(main_data_dir, model_arch, dataset_config, dataset_mode)
    lmdb_dataset = CustomLMDBDataset(lmdb_path, csv_path)
    if data_idx > len(lmdb_dataset):
        raise ValueError(f'Index is out of bounds. Dataset has only {len(lmdb_dataset)} samples.')
    
    img_tensor, img_label, img_name = lmdb_dataset.get_one_image(data_idx)
    if len(img_tensor.shape)==3:
        img_tensor = img_tensor.unsqueeze(0)

    # Get model prediction.
    model = model.to(device).eval()
    img_tensor = img_tensor.to(device)
    output = model(img_tensor)
    _, pred = torch.max(output, 1)
    pred_label = pred.item()

    # Generate and visualise Guided Grad-CAM saliency map.
    guided_gradcam = CustomGuidedGradCam(model, target_layer, device=device)
    guided_gradcam_map = guided_gradcam.generate_map(img_tensor, target_class, skew=skew)

    img_ori_path = os.path.join(main_data_dir, "processed-data", dataset_config, dataset_mode, "png-files", img_name)
    img_ori = Image.open(img_ori_path).convert('RGB')
    img_ori_tensor = transforms.functional.pil_to_tensor(img_ori)
    guided_gradcam_img = show_cam_on_image(img_ori_tensor, guided_gradcam_map, alpha=adj_viz_alpha, adj_bright=adj_viz_bright)

    return guided_gradcam_img, img_label, pred_label, img_name
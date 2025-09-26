import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from inspect import signature



# --------------------------- #
# Factory Function: Optimizer #
# --------------------------- #
# > To be adjusted if hyperparameter values include additional algorithms.
def make_optimizer_fn(optimizerName, learningRate, weightDecay, momentum=0.9):
    """
    Factory function to create a function that instantiates a PyTorch optimizer. Currently,
    only supports Adam, AdamW, and SGD. Only learning rate and weight decay parameters
    are allowed to be parsed, except for SGD which also accepts momentum.

    Arguments:
        optimizerName (str): Name of the optimizer. (e.g., Adam)
        learningRate (float): Learning rate value.
        weightDecay (float): Weight decay value.
        momentum (float, optional): Momentum value. Only relevant for SGD. (Default: 0.9)
    
    Returns:
         function: A function that takes a PyTorch model as input and returns an instance of the specified optimizer.
    """
    def optimizer_fn(model):
        """
        Creates and returns a PyTorch optimizer instance for a given model.

        Arguments:
            model (torch.nn.Module): The PyTorch model for which to create the optimizer.

        Returns:
            torch.optim.Optimizer: An instantiated PyTorch optimizer.
        """
        # Remove frozen params to reduce memory requirements.
        params = filter(lambda p: p.requires_grad, model.parameters())
        
        # Return relevant optimizer.
        if optimizerName == 'Adam':
            return optim.Adam(params, lr=learningRate, weight_decay=weightDecay)
        elif optimizerName == 'SGD':
            return optim.SGD(params, lr=learningRate, weight_decay=weightDecay, momentum=momentum)
        elif optimizerName == 'AdamW':
            return optim.AdamW(params, lr=learningRate, weight_decay=weightDecay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizerName}. Consider adding the option into the source function.")
    
    # Return optimizer function.
    return optimizer_fn



# --------------------------- #
# Factory Function: Scheduler #
# --------------------------- #
# > To be adjusted if hyperparameter values include additional algorithms.
def make_scheduler_fn(
        sched_name,
        stepSize=20, gammaSize=0.5,
        metricMode='max', reduceFactor=0.1, reducePatience=5,
        **kwargs
):
    """
    Factory function to create a function that instantiates a PyTorch scheduler. Currently,
    only supports StepLR, ReduceLROnPlateau, and OneCycleLR. Any parameters for each
    of the scheduler can be passed through and will be inherited by the relevant schedulers.

    Arguments:
        sched_name (str): Name of scheduler. (e.g., StepLR)
        stepSize (int, optional): Number of steps before learning rate is reduced. Only relevant for StepLR. (Default: 20)
        gammaSize (float, optional): Size of learning rate to be reduced. Only relevant for StepLR. (Default: 0.5)
        metricMode (str, optiona): Select between min or max monitoring mode. Only relevant for ReduceLROnPlateau. (Default: max)
        reduceFactor (float, optional): Factor by which learning rate will be reduced. Only relevant for ReduceLROnPlateau. (Default: 0.1)
        reducePatience (int, optional): The number of allowed epochs with no improvement after which the learning rate will be reduced. (Default: 5)
        **kwargs: Other relevant parameters for the corrrsponding schedulers.
    
    Returns:
        function: A function that takes a PyTorch optimizer as input and returns an instance of the specified scheduler.
    """
    # Return relevant scheduler.
    def scheduler_fn(optimizer):
        """
        Creates and returns a PyTorch scheduler instance for a given model.

        Arguments:
            optimizer (torch.optim.Optimizer): The PyTorch optimizer that handles the learning rate.

        Returns:
            torch.optim.lr_scheduler: An instantiated PyTorch scheduler.
        """
        if sched_name=="None":
            return None
        
        elif sched_name=='StepLR':
            valid_keys = signature(optim.lr_scheduler.StepLR).parameters
            sched_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            return optim.lr_scheduler.StepLR(optimizer, step_size=stepSize, gamma=gammaSize, **sched_kwargs)
        
        elif sched_name=='ReduceLROnPlateau':
            valid_keys = signature(optim.lr_scheduler.ReduceLROnPlateau).parameters
            sched_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=metricMode, factor=reduceFactor, patience=reducePatience, **sched_kwargs)
        
        elif sched_name=='OneCycleLR':
            valid_keys = signature(optim.lr_scheduler.OneCycleLR).parameters
            sched_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            return optim.lr_scheduler.OneCycleLR(optimizer, **sched_kwargs)
        
        else:
            raise ValueError(f"Unsupported scheduler: {sched_name}. Consider adding the option into the source function.")

    # Return scheduler function.  
    return scheduler_fn



# --------------------------- #
# Factory Function: CNN Model #
# --------------------------- #
# > To be adjusted if experimentations include additional layer intialisation methods.
def apply_layer_init(modelLayer, initMethod):
    """
    Applies the corresponding weight initialisation method to the newly initialised PyTorch layer.

    Arguments:
        modelLayer (torch.nn.Module): The newly instantiated layer.
        initMethod (str): Initialisation method. If None, the default method is used.
    """
    if initMethod=="None":
        return  None      # Do nothing. Use default initialisation method.
    
    # Map string names to actual init functions.
    init_methods = {
        'xavier_uniform': nn.init.xavier_uniform_,
        'xavier_normal': nn.init.xavier_normal_,
        'kaiming_uniform': nn.init.kaiming_uniform_,
        'kaiming_normal': nn.init.kaiming_normal_,
        'normal': nn.init.normal_,
        'uniform': nn.init.uniform_,
        'zeros': lambda x: nn.init.constant_(x, 0)
    }

    if initMethod not in init_methods:
        raise ValueError(f"Unsupported init method: {initMethod}. Consider adding the option into the source function.")

    init_fn = init_methods[initMethod]
    if hasattr(modelLayer, 'weight'):
        init_fn(modelLayer.weight)



# > To be adjusted if experimentations include additional architectures.
def get_cnn_model(architectName, numClasses, freezeBackbone=True, initMethod="None", stateDictPath=None):
    """
    Factory function to instantiate a torchvision model, with pre-trained ImageNet-1K weights. Replaces
    the classifier head with a new layer that and has numClasses number of outputs. Currently
    supports resnet18, resnet50, mobilenetv3small, mobilenetv3large, efficientnetb0, efficientb3,
    convnexttiny, convnextbase, and vitb16.

    Arguments:
        architectName (str): Name of model.
        numClasses (int): Number of unique labels to predict.
        freezeBackbone (bool, optional): If true, only train classifier head. Otherwise, all layers are trainable. (Default: True)
        initMethod (str, optional): Weight initialisation method for classifier head. If "None", use the default. (Default: None)
        stateDictPath (str, optional): Path to state dict to load previously trained weights. (Default: None)
    
    Returns:
        torchvision.models.Model: An instantiated PyTorch model.
    """
    architectName = architectName.lower()

    if architectName == 'resnet18':
        # Load pre-trained weights.
        cnn_model = models.resnet18(weights='IMAGENET1K_V1')
        
        if freezeBackbone:
            # Freeze all exisiting layers for transfer learning.
            print("Training mode: transfer learning. Backbone layers are frozen, only train classification head.")
            for param in cnn_model.parameters():
                param.requires_grad = False
            
            # Replace classification head.
            numFeatures = cnn_model.fc.in_features
            cnn_model.fc = nn.Linear(numFeatures, numClasses)
            apply_layer_init(cnn_model.fc, initMethod)
            print(f'Replaced classification head with intialisation method: {initMethod}')
        
        else:
            # Fine tune the whole model.
            print("Training mode: fine tuning. All layers are trainable.")
            numFeatures = cnn_model.fc.in_features
            cnn_model.fc = nn.Linear(numFeatures, numClasses)

            # Load saved parameters.
            if stateDictPath is not None:
                try:
                    savedParams = torch.load(stateDictPath, map_location='cpu', weights_only=True)
                    cnn_model.load_state_dict(savedParams)
                    print(f'Successfully instantiated model with saved params from: {stateDictPath}')
                except Exception as e:
                    print(f'[ERROR!] Error loading state dictionary from {stateDictPath}: {e}.')
                    raise


    elif architectName == 'resnet50':
        # Load pre-trained weights.
        cnn_model = models.resnet50(weights='IMAGENET1K_V2')
        
        if freezeBackbone:
            # Freeze all exisiting layers for transfer learning.
            print("Training mode: transfer learning. Backbone layers are frozen, only train classification head.")
            for param in cnn_model.parameters():
                param.requires_grad = False
            
            # Replace classification head.
            numFeatures = cnn_model.fc.in_features
            cnn_model.fc = nn.Linear(numFeatures, numClasses)
            apply_layer_init(cnn_model.fc, initMethod)
            print(f'Replaced classification head with intialisation method: {initMethod}')
        
        else:
            # Fine tune the whole model.
            print("Training mode: fine tuning. All layers are trainable.")
            numFeatures = cnn_model.fc.in_features
            cnn_model.fc = nn.Linear(numFeatures, numClasses)

            # Load saved parameters.
            if stateDictPath is not None:
                try:
                    savedParams = torch.load(stateDictPath, map_location='cpu', weights_only=True)
                    cnn_model.load_state_dict(savedParams)
                    print(f'Successfully instantiated model with saved params from: {stateDictPath}')
                except Exception as e:
                    print(f'[ERROR!] Error loading state dictionary from {stateDictPath}: {e}.')
                    raise
    

    elif architectName == "efficientnetb0":
        # Load pre-trained weights.
        cnn_model = models.efficientnet_b0(weights='IMAGENET1K_V1')

        if freezeBackbone:
            # Freeze all exisiting layers for transfer learning.
            print("Training mode: transfer learning. Backbone layers are frozen, only train classification head.")
            for param in cnn_model.parameters():
                param.requires_grad = False
            
            # Replace classification head.
            numFeatures = cnn_model.classifier[-1].in_features
            cnn_model.classifier[-1] = nn.Linear(numFeatures, numClasses)
            apply_layer_init(cnn_model.classifier[-1], initMethod)
            print(f'Replaced classification head with intialisation method: {initMethod}')
        
        else:
            # Fine tune the whole model.
            print("Training mode: fine tuning. All layers are trainable.")
            numFeatures = cnn_model.classifier[-1].in_features
            cnn_model.classifier[-1] = nn.Linear(numFeatures, numClasses)

            # Load saved parameters.
            if stateDictPath is not None:
                try:
                    savedParams = torch.load(stateDictPath, map_location='cpu', weights_only=True)
                    cnn_model.load_state_dict(savedParams)
                    print(f'Successfully instantiated model with saved params from: {stateDictPath}')
                except Exception as e:
                    print(f'[ERROR!] Error loading state dictionary from {stateDictPath}: {e}.')
                    raise
    

    elif architectName == "efficientnetb3":
        # Load pre-trained weights.
        cnn_model = models.efficientnet_b3(weights='IMAGENET1K_V1')

        if freezeBackbone:
            # Freeze all exisiting layers for transfer learning.
            print("Training mode: transfer learning. Backbone layers are frozen, only train classification head.")
            for param in cnn_model.parameters():
                param.requires_grad = False
            
            # Replace classification head.
            numFeatures = cnn_model.classifier[-1].in_features
            cnn_model.classifier[-1] = nn.Linear(numFeatures, numClasses)
            apply_layer_init(cnn_model.classifier[-1], initMethod)
            print(f'Replaced classification head with intialisation method: {initMethod}')
        
        else:
            # Fine tune the whole model.
            print("Training mode: fine tuning. All layers are trainable.")
            numFeatures = cnn_model.classifier[-1].in_features
            cnn_model.classifier[-1] = nn.Linear(numFeatures, numClasses)

            # Load saved parameters.
            if stateDictPath is not None:
                try:
                    savedParams = torch.load(stateDictPath, map_location='cpu', weights_only=True)
                    cnn_model.load_state_dict(savedParams)
                    print(f'Successfully instantiated model with saved params from: {stateDictPath}')
                except Exception as e:
                    print(f'[ERROR!] Error loading state dictionary from {stateDictPath}: {e}.')
                    raise
    

    elif architectName == 'vitb16':
        # Load pre-trained weights.
        cnn_model = models.vit_b_16(weights='IMAGENET1K_V1')

        if freezeBackbone:
            # Freeze all exisiting layers for transfer learning.
            print("Training mode: transfer learning. Backbone layers are frozen, only train classification head.")
            for param in cnn_model.parameters():
                param.requires_grad = False
            
            # Replace classification head.
            numFeatures = cnn_model.heads[-1].in_features
            cnn_model.heads[-1] = nn.Linear(numFeatures, numClasses)
            apply_layer_init(cnn_model.heads[-1], initMethod)
            print(f'Replaced classification head with intialisation method: {initMethod}')
        
        else:
            # Fine tune the whole model.
            print("Training mode: fine tuning. All layers are trainable.")
            numFeatures = cnn_model.heads[-1].in_features
            cnn_model.heads[-1] = nn.Linear(numFeatures, numClasses)

            # Load saved parameters.
            if stateDictPath is not None:
                try:
                    savedParams = torch.load(stateDictPath, map_location='cpu', weights_only=True)
                    cnn_model.load_state_dict(savedParams)
                    print(f'Successfully instantiated model with saved params from: {stateDictPath}')
                except Exception as e:
                    print(f'[ERROR!] Error loading state dictionary from {stateDictPath}: {e}.')
                    raise
    

    elif architectName == 'convnexttiny':
        # Load pre-trained weights.
        cnn_model = models.convnext_tiny(weights='IMAGENET1K_V1')

        if freezeBackbone:
            # Freeze all exisiting layers for transfer learning.
            print("Training mode: transfer learning. Backbone layers are frozen, only train classification head.")
            for param in cnn_model.parameters():
                param.requires_grad = False
            
            # Replace classification head.
            numFeatures = cnn_model.classifier[-1].in_features
            cnn_model.classifier[-1] = nn.Linear(numFeatures, numClasses)
            apply_layer_init(cnn_model.classifier[-1], initMethod)
            print(f'Replaced classification head with intialisation method: {initMethod}')
        
        else:
            # Fine tune the whole model.
            print("Training mode: fine tuning. All layers are trainable.")
            numFeatures = cnn_model.classifier[-1].in_features
            cnn_model.classifier[-1] = nn.Linear(numFeatures, numClasses)

            # Load saved parameters.
            if stateDictPath is not None:
                try:
                    savedParams = torch.load(stateDictPath, map_location='cpu', weights_only=True)
                    cnn_model.load_state_dict(savedParams)
                    print(f'Successfully instantiated model with saved params from: {stateDictPath}')
                except Exception as e:
                    print(f'[ERROR!] Error loading state dictionary from {stateDictPath}: {e}.')
                    raise
    

    elif architectName == 'convnextbase':
        # Load pre-trained weights.
        cnn_model = models.convnext_base(weights='IMAGENET1K_V1')

        if freezeBackbone:
            # Freeze all exisiting layers for transfer learning.
            print("Training mode: transfer learning. Backbone layers are frozen, only train classification head.")
            for param in cnn_model.parameters():
                param.requires_grad = False
            
            # Replace classification head.
            numFeatures = cnn_model.classifier[-1].in_features
            cnn_model.classifier[-1] = nn.Linear(numFeatures, numClasses)
            apply_layer_init(cnn_model.classifier[-1], initMethod)
            print(f'Replaced classification head with intialisation method: {initMethod}')
        
        else:
            # Fine tune the whole model.
            print("Training mode: fine tuning. All layers are trainable.")
            numFeatures = cnn_model.classifier[-1].in_features
            cnn_model.classifier[-1] = nn.Linear(numFeatures, numClasses)

            # Load saved parameters.
            if stateDictPath is not None:
                try:
                    savedParams = torch.load(stateDictPath, map_location='cpu', weights_only=True)
                    cnn_model.load_state_dict(savedParams)
                    print(f'Successfully instantiated model with saved params from: {stateDictPath}')
                except Exception as e:
                    print(f'[ERROR!] Error loading state dictionary from {stateDictPath}: {e}.')
                    raise
    

    elif architectName == 'mobilenetv3large':
        # Load pre-trained weights.
        cnn_model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')

        if freezeBackbone:
            # Freeze all exisiting layers for transfer learning.
            print("Training mode: transfer learning. Backbone layers are frozen, only train classification head.")
            for param in cnn_model.parameters():
                param.requires_grad = False
            
            # Replace classification head.
            numFeatures = cnn_model.classifier[-1].in_features
            cnn_model.classifier[-1] = nn.Linear(numFeatures, numClasses)
            apply_layer_init(cnn_model.classifier[-1], initMethod)
            print(f'Replaced classification head with intialisation method: {initMethod}')
        
        else:
            # Fine tune the whole model.
            print("Training mode: fine tuning. All layers are trainable.")
            numFeatures = cnn_model.classifier[-1].in_features
            cnn_model.classifier[-1] = nn.Linear(numFeatures, numClasses)

            # Load saved parameters.
            if stateDictPath is not None:
                try:
                    savedParams = torch.load(stateDictPath, map_location='cpu', weights_only=True)
                    cnn_model.load_state_dict(savedParams)
                    print(f'Successfully instantiated model with saved params from: {stateDictPath}')
                except Exception as e:
                    print(f'[ERROR!] Error loading state dictionary from {stateDictPath}: {e}.')
                    raise
    

    elif architectName == 'mobilenetv3small':
        # Load pre-trained weights.
        cnn_model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')

        if freezeBackbone:
            # Freeze all exisiting layers for transfer learning.
            print("Training mode: transfer learning. Backbone layers are frozen, only train classification head.")
            for param in cnn_model.parameters():
                param.requires_grad = False
            
            # Replace classification head.
            numFeatures = cnn_model.classifier[-1].in_features
            cnn_model.classifier[-1] = nn.Linear(numFeatures, numClasses)
            apply_layer_init(cnn_model.classifier[-1], initMethod)
            print(f'Replaced classification head with intialisation method: {initMethod}')
        
        else:
            # Fine tune the whole model.
            print("Training mode: fine tuning. All layers are trainable.")
            numFeatures = cnn_model.classifier[-1].in_features
            cnn_model.classifier[-1] = nn.Linear(numFeatures, numClasses)

            # Load saved parameters.
            if stateDictPath is not None:
                try:
                    savedParams = torch.load(stateDictPath, map_location='cpu', weights_only=True)
                    cnn_model.load_state_dict(savedParams)
                    print(f'Successfully instantiated model with saved params from: {stateDictPath}')
                except Exception as e:
                    print(f'[ERROR!] Error loading state dictionary from {stateDictPath}: {e}.')
                    raise


    else:
        raise ValueError(f"Unsupported architecture: {architectName}. Consider adding the option into the source function.")
    
    # Returns CNN model.
    print(f'Model configuration complete for {architectName}\n')
    return cnn_model

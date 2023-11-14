from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights 
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights


def load_model(config):
    """
    Load models based on the configuration file.

    Args:
        config (dict): A dictionary containing the configuration for the models.

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary contains the loaded models, and the second dictionary contains the transformations for each model.
    """
    models = {}
    transformations = {}
    for model_name in config.models:
        if model_name == 'keypoints':
            weights = config.models.keypoints.get("weights")
            transformations['keypoints'] = weights.transforms()
            model = keypointrcnn_resnet50_fpn(weights=weights, progress=False)
            model = model.eval()
            models['keypoints'] = model

        if model_name == 'semantic_segmentation':
            weights = config.models.semantic_segmentation.weights
            transformations['semantic_segmentation'] = weights.transforms()
            model =  fcn_resnet101(weights)
            model = model.eval()
            models['semantic_segmentation'] = model

        
    return models, transformations

import importlib

def load_model(config):
    models = {}
    transformations = {}
    for model_name, model_info in config.models.items():
        # Dynamically import the model function
        model_func_module, model_func_name = model_info.model_loading_function.rsplit('.', 1)
        model_func = getattr(importlib.import_module(model_func_module), model_func_name)

        # Dynamically import the weights
        weights_module, weights_class_name = model_info.weights.rsplit('.', 1)
        weights_class = getattr(importlib.import_module(weights_module), weights_class_name)
        weights = getattr(weights_class, model_info.weight_version)

        transformations[model_name] = weights.transforms()
        model = model_func(weights=weights, progress=False)
        model = model.eval()
        models[model_name] = model

    return models, transformations
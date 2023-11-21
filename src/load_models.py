from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights 
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights

import importlib

def load_model_non_hydra(config):
    models = {}
    transformations = {}
    for task_type, task in config.tasks.items():
        models[task_type] = {}
        transformations[task_type] = {}
        for model_name, model_config in task.items():
            # Dynamically import the model loading function
            module_name, function_name = model_config.load_model.model_loading_function.rsplit('.', 1)
            module = importlib.import_module(module_name)
            model_loading_function = getattr(module, function_name)

            # Dynamically import the weights
            weights_module, weights_class_name = model_config.load_model.weights.rsplit('.', 1)
            weights_class = getattr(importlib.import_module(weights_module), weights_class_name)
            weights = getattr(weights_class, model_config.load_model.weight_version)
            # Load the model
            model = model_loading_function(weights=weights)

            # Load the transformations
            transformation = weights.transforms()
            transformations[task_type][model_name] = transformation

            models[task_type][model_name] = model

    return models, transformations
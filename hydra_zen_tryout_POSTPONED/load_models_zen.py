import importlib
import torch


def load_model(model_loading_function: str, weights: str, weight_version: str):
    # Dynamically import the model loading function
    module_name, function_name = model_loading_function.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_loading_function = getattr(module, function_name)

    # Dynamically import the weights
    weights_module, weights_class_name = weights.rsplit('.', 1)
    weights_class = getattr(importlib.import_module(weights_module), weights_class_name)
    weights = getattr(weights_class, weight_version)

    # Load the model
    model = model_loading_function(weights=weights)

    # Load the transformations
    transformations = weights.transforms()

    return model, transformations


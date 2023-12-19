from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights 
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights
import torch 
import src.transformations as transformation_functions
import importlib
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
import os
import json

def load_model_torchhub(config_load_model):

    model_variant = config_load_model.model_variant

    model_repo = config_load_model.model_repo

    if config_load_model.pretrained:
        model = torch.hub.load(model_repo, model_variant, pretrained=True)
        return model, model_variant


def load_model_non_hydra(config):
    models = {}
    transformations = {}
    classes = {}
    clip_durations = {}
    for task_type, task in config.tasks.items():
        models[task_type] = {}
        transformations[task_type] = {}
        classes[task_type] = {}
        clip_durations[task_type] = {}
        for model_name, model_config in task.items():
            if model_config.framework == 'torch':
                # Dynamically import the model loading function
                module_name, function_name = model_config.load_model.model_loading_function.rsplit('.', 1)
                module = importlib.import_module(module_name)
                model_loading_function = getattr(module, function_name)

                # Dynamically import the weights
                weights_module, weights_class_name = model_config.load_model.weights.rsplit('.', 1)
                weights_class = getattr(importlib.import_module(weights_module), weights_class_name)
                weights = getattr(weights_class, model_config.load_model.weight_version)
                classes_torch  = weights.meta['categories']
                # Load the model
                model = model_loading_function(weights=weights)

                # Load the transformations
                transformation = weights.transforms()
                transformations[task_type][model_name] = transformation
                models[task_type][model_name] = model
                classes[task_type][model_name] = classes_torch
            if model_config.framework == 'torchhub':
                model, model_variant = load_model_torchhub(model_config.load_model)
                models[task_type][model_name] = model
                

                transformation, clip_duration = transformation_functions.torch_transform(model_variant, config)
                transformations[task_type][model_name] = transformation
                clip_durations[task_type][model_name] = clip_duration
                dataset_classes = model_config.load_model.get('classes') 
                print(dataset_classes)
                classes_file_path = config.class_paths[dataset_classes]
                with open(classes_file_path, 'r') as f:
                    kinetics_classnames= json.load(f)
                kinetics_id_to_classname = {}
                for k, v in kinetics_classnames.items():
                    kinetics_id_to_classname[v] = str(k).replace('"', "")
                classes[task_type][model_name] = kinetics_id_to_classname
            
            if model_config.framework == 'pytorchvideo':
                print('here')
                model, model_variant = load_model_torchhub(model_config.load_model)
                models[task_type][model_name] = model
                dataset_classes = model_config.load_model.get('classes')
                classes_file_path = config.class_paths[dataset_classes]
                label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map(classes_file_path)
                print(label_map)
                print(allowed_class_ids)
                classes[task_type][model_name] = [[label_map],[allowed_class_ids]]
                transformation, clip_duration = transformation_functions.torch_transform(model_variant, config)
                if transformation == None:
                    print('no transformation')
                transformations[task_type][model_name] = transformation
                clip_durations[task_type][model_name] = clip_duration
    return models, transformations, classes, clip_durations
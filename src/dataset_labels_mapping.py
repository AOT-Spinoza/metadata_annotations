import json
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from torchvision.models import get_weight

def classes_mapping(parameters_classes, config):
    if 'weights' in parameters_classes:
        weights = parameters_classes['weights']
        weights = get_weight(weights)
        classes = weights.meta["categories"]
        return classes
    elif 'dataset' in parameters_classes:
        dataset_trained_on = parameters_classes['dataset']
        classes_file_path = config.class_paths[dataset_trained_on]
    
        if dataset_trained_on == "kinetics400":
            ##Action classfication
            with open(classes_file_path, 'r') as f:
                kinetics_classnames = json.load(f)
    
            # Map kinetics IDs to class names
            kinetics_id_to_classname = {}
            for k, v in kinetics_classnames.items():
                kinetics_id_to_classname[v] = str(k).replace('"', "")
    
            return kinetics_id_to_classname
        elif dataset_trained_on == "ava":

            ##Action Detection
            label_map, _ = AvaLabeledVideoFramePaths.read_label_map(classes_file_path)
            return label_map
    else:
        raise FileNotFoundError(f"Dataset {parameters_classes.dataset} not found in {classes_file_path}")
    
    
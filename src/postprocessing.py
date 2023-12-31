import yaml
import importlib
from src.tracker import tracking
import torch
import numpy as np

def threshold(predictions, threshold_value=0.75):
    """
    Applies a threshold to the predictions of the Keypoint R-CNN model.

    Args:
        prediction (dict): A dictionary containing the predictions made by the model.
        threshold_value (float): The threshold value.

    Returns:
        dict: A dictionary containing the filtered predictions.
    """
    filtered_predictions = []
    for prediction in predictions:
        scores = prediction['scores']
        idx = torch.where(scores > float(threshold_value))

        # Filter all keys in the prediction using idx
        filtered_prediction = {key: value[idx] for key, value in prediction.items() if isinstance(value, torch.Tensor)}
        filtered_predictions.append(filtered_prediction)

    return filtered_predictions

def soft_max(predictions, config):
    """
    Applies the softmax function to the predictions of the semantic segmentation model.

    Args:
        predictions (list of Tensors): A list of tensors containing the predictions made by the model.

    Returns:
        list of Tensors: A list of tensors containing the predictions with applied softmax function.
    """
    print('soft_max')
    return [torch.nn.functional.softmax(prediction.squeeze(), dim=0) for prediction in predictions]





def postprocess_predictions(predictions, config):
    """
    Applies post-processing functions to the predictions made by the models.

    Args:
        predictions (dict): A dictionary containing the predictions made by the models.
        config (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing the post-processed predictions.
    """
    postprocessed = {}
    for task_type, task_results in predictions.items():
        postprocessed[task_type] = {}
        for model_name, videos in task_results.items():
            if config['tasks'][task_type][model_name].get('postprocessing', None) == None:
                postprocessed[task_type][model_name] = videos
                continue
            postprocessing_func_names = config['tasks'][task_type][model_name]['postprocessing']
            postprocessed[task_type][model_name] = {}
            for video_name, prediction in videos.items():
                postprocessed[task_type][model_name][video_name] = prediction
                for postprocessing_func in postprocessing_func_names:
                    # Dynamically import the post-processing function
                    module_name, function_name = postprocessing_func.rsplit('.', 1)
                    module = importlib.import_module(module_name)
                    postprocessing_function = getattr(module, function_name)
                    
                    # Apply the post-processing function
                    if function_name == 'threshold':
                        threshold_value = config['tasks'][task_type][model_name].get('threshold_value', 0.75)  # Use a default value if not specified
                        postprocessed[task_type][model_name][video_name] = postprocessing_function(postprocessed[task_type][model_name][video_name], threshold_value)
                    else:
                        # print(f"Before {function_name}, shape: {postprocessed[task_type][model_name][video_name][0].shape}")
                        print('postprocessing')
                        print(model_name)
                        postprocessed[task_type][model_name][video_name] = postprocessing_function(postprocessed[task_type][model_name][video_name], config)
                        #  print(f"After {function_name}, shape: {postprocessed[task_type][model_name][video_name][0].shape}")
                        print('done')
    return postprocessed
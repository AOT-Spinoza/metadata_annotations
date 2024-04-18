import yaml
import importlib
from src.tracker import tracking
import torch
import numpy as np

def check_for_persons(predictions, config, video_name=None):
    """
    Checks if there are any persons detected in the predictions.

    Args:
        predictions (dict): A dictionary containing the predictions made by the model.

    Returns:
        bool: True if there are persons detected, False otherwise.
    """
    lower_limit = config['tasks']['keypoints']['KeypointRCNN_ResNet50']['lower_limit_persons']
    frames_with_persons = 0
    frames_with_no_persons = 0
    for prediction in predictions:
        if len(prediction['boxes']) > 0:
            frames_with_persons += 1
        else:
            frames_with_no_persons += 1

    if frames_with_persons >= lower_limit:
        # if frames_with_persons > frames_with_no_persons:
        #     print('manypersonsdetected')
        return predictions
    
    return None

def threshold(predictions, threshold_value=0.75, video_name=None):
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

def soft_max(predictions, config, video_name):
    """
    Applies the softmax function to the predictions of the semantic segmentation model.

    Args:
        predictions (list of Tensors): A list of tensors containing the predictions made by the model.

    Returns:
        list of Tensors: A list of tensors containing the predictions with applied softmax function.
    """

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
    # Initialize an empty dictionary to store the post-processed predictions
    postprocessed = {}

    # Loop over each task type and its corresponding results in the predictions
    for task_type, task_results in predictions.items():
        postprocessed[task_type] = {}

        # Loop over each model and its corresponding videos in the task results
        for model_name, videos in task_results.items():

            # If no post-processing is specified for this model, copy the videos as is
            if config['tasks'][task_type][model_name].get('postprocessing', None) == None:
                postprocessed[task_type][model_name] = videos
                continue

            # Get the list of post-processing functions for this model
            postprocessing_func_names = config['tasks'][task_type][model_name]['postprocessing']
            postprocessed[task_type][model_name] = {}

            # Loop over each video and its corresponding prediction in the videos
            for video_name, prediction in videos.items():
                postprocessed[task_type][model_name][video_name] = prediction

                # Loop over each post-processing function
                for postprocessing_func in postprocessing_func_names:


                    # Dynamically import the post-processing function
                    module_name, function_name = postprocessing_func.rsplit('.', 1)
                    module = importlib.import_module(module_name)
                    postprocessing_function = getattr(module, function_name)
                    
                    # Apply the post-processing function
                    if function_name == 'threshold':
                        # If the function is 'threshold', get the threshold value from the config (default to 0.75 if not specified)
                        threshold_value = config['tasks'][task_type][model_name].get('threshold_value', 0.75)
                        postprocessed[task_type][model_name][video_name] = postprocessing_function(postprocessed[task_type][model_name][video_name], threshold_value)
                    else:
                        # For other functions, pass the prediction and the config as arguments
                        postprocessed[task_type][model_name][video_name] = postprocessing_function(postprocessed[task_type][model_name][video_name], config, video_name)

    # Return the post-processed predictions
    return postprocessed
import yaml
import importlib
from src.tracker import track_boxes
import torch


def threshold(prediction, threshold_value=0.75):
    """
    Applies a threshold to the predictions of the Keypoint R-CNN model.

    Args:
        prediction (dict): A dictionary containing the predictions made by the model.
        threshold_value (float): The threshold value.

    Returns:
        dict: A dictionary containing the filtered predictions.
    """
    scores = prediction['scores']
    idx = torch.where(scores > threshold_value)
    filtered_prediction = prediction[idx]
    return filtered_prediction

def soft_max(prediction):
    """
    Applies the softmax function to the predictions of the semantic segmentation model.

    Args:
        prediction (Tensor): A tensor containing the predictions made by the model.

    Returns:
        Tensor: A tensor containing the predictions with applied softmax function.
    """
    return torch.nn.functional.softmax(prediction, dim=0)


def tracking(predictions, config):
    ## track the predictions through frames, instances getting unique ids, with the use of the sort function
    ## TO BE IMPLEMENTED

    return predictions



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
            postprocessing_func_names = config['tasks'][task_type][model_name]['postprocessing']
            postprocessed[task_type][model_name] = {}
            for video_name, prediction in videos.items():
                for postprocessing_func in postprocessing_func_names:
                    # Dynamically import the post-processing function
                    module_name, function_name = postprocessing_func['function'].rsplit('.', 1)
                    module = importlib.import_module(module_name)
                    postprocessing_function = getattr(module, function_name)
                    # Apply the post-processing function
                    if function_name == 'threshold':
                        threshold_value = postprocessing_func.get('threshold_value', 0.75)  # Use a default value if not specified
                        postprocessed_prediction = postprocessing_function(prediction, threshold_value)
                    else:
                        postprocessed_prediction = postprocessing_function(prediction)
                postprocessed[task_type][model_name][video_name] = postprocessed_prediction
    return postprocessed
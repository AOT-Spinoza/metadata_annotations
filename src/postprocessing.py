import importlib
import yaml

def postprocessing(config, predictions):
    """
    Postprocess the predictions.

    Args:
        config: The configuration YAML file.
        predictions (dict): Dictionary containing the predictions.

    Returns:
        dict: Dictionary containing the postprocessed predictions.
    """

    postprocessed = {}
    for model_name, prediction in predictions.items():
        postprocessing_func_names = config['models'][model_name]['postprocessing']
        postprocessing_module_name = f"src.postprocessing.{model_name}"
        postprocessing_module = importlib.import_module(postprocessing_module_name)
        postprocessing_func = getattr(postprocessing_module, postprocessing_func_name)
        postprocessed[model_name] = postprocessing_func(prediction, config)
    return postprocessed

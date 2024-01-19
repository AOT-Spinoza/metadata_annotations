import importlib
from transformers import AutoProcessor, AutoModelForCausalLM

def import_from(dotted_path):
    """
    Import a function or class from a dotted path.

    Args:
        dotted_path (str): The dotted path of the function or class to import.

    Returns:
        function or class: The imported function or class.

    Raises:
        ImportError: If the module or attribute does not exist.
    """
    parts = dotted_path.split('.')
    if parts[-1] == 'from_pretrained':
        module_name, class_name = '.'.join(parts[:-2]), parts[-2]
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        return getattr(class_, 'from_pretrained')
    else:
        module_name, function_name = '.'.join(parts[:-1]), parts[-1]
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        return function


def load_and_configure_model(model_config, config):
    """
    Loads a model from Torch Hub and performs necessary transformations.

    Args:
        model_config (ModelConfig): Configuration object for the model.
        config (Config): Configuration object for the application.

    Returns:
        tuple: A tuple containing the loaded model, transformation function,
               dictionary mapping kinetics IDs to class names, and clip duration.
    """
    # Extract model configuration parameters
    framework = model_config.framework

    load_function = import_from(model_config.model_loading_function)

    # Load the model

    model = load_function(**model_config.parameters_load_function)

    # Perform necessary transformations
    transformation_function = import_from(model_config.transformation_function)
    transformation, clip_duration = transformation_function(**model_config.parameters_transformation)

    # Load the class names from the specified file
    try:

        classes_function = import_from(model_config.classes_function)
        classes_map = classes_function(model_config.parameters_classes, config)

    except:
        classes_map = None

    # Return the loaded model, transformation function, class name mapping, and clip duration
    return model, transformation, classes_map, clip_duration

def load_all_models_and_metadata(config):
    """
    Load all models and metadata based on the provided configuration.

    Args:
        config (dict): The configuration containing information about the models to load.

    Returns:
        tuple: A tuple containing dictionaries of models, transformations, classes, and clip durations.
            - models (dict): A nested dictionary containing the loaded models.
            - transformations (dict): A nested dictionary containing the transformations for each model.
            - classes (dict): A nested dictionary containing the class lists for each model.
            - clip_durations (dict): A nested dictionary containing the clip durations for each model.
    """
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
            # Dynamically import the model loading function
            model, transformation, class_list, clip_duration = load_and_configure_model(model_config.load_model, config)
            transformations[task_type][model_name] = transformation
            models[task_type][model_name] = model
            classes[task_type][model_name] = class_list
            clip_durations[task_type][model_name] = clip_duration


    return models, transformations, classes, clip_durations

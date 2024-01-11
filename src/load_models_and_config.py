import importlib


def import_from(dotted_path):
    module_name, function_name = dotted_path.rsplit('.', 1)
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
    print(model_config.parameters_transformation)
    load_function = import_from(model_config.model_loading_function)

    # Load the model
    if framework == "torchhub" or framework == "pytorchvideo":
        model = load_function(**model_config.parameters_load_function)
    elif framework == "torch":
        # Dynamically import the weights for the current model
        # print(model_config.parameters_load_function.weights)
        # weights = import_from(model_config.parameters_load_function.weights)
        # model_config.parameters_load_function['weights'] = weights
        # model_config.parameters_transformation['weights'] = weights
        # model_config.parameters_classes['weights'] = weights
        model = load_function(**model_config.parameters_load_function)
        print('success')
    print(model_config.parameters_transformation)
    # Perform necessary transformations
    transformation_function = import_from(model_config.transformation_function)
    transformation, clip_duration = transformation_function(**model_config.parameters_transformation, config =config)

    # Load the class names from the specified file
    try:
        classes_function = import_from(model_config.classes_function)
        classes_map = classes_function(model_config.parameters_classes)
    except:
        classes_map = None

    # Return the loaded model, transformation function, class name mapping, and clip duration
    return model, transformation, classes_map, clip_duration

def load_all_models_and_metadata(config):
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
            print(model_name)
            # Dynamically import the model loading function
            model, transformation, class_list, clip_duration = load_and_configure_model(model_config.load_model, config)
            transformations[task_type][model_name] = transformation
            models[task_type][model_name] = model
            print(classes)
            classes[task_type][model_name] = class_list
            clip_durations[task_type][model_name] = clip_duration

    return models, transformations, classes, clip_durations

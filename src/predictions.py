from scripts import torch_inference
import os

def inference(config, models, transformations, input_dir):
    """
    Runs inference on the given video files using the specified models and transformations.

    Args:
        config (dict): Configuration dictionary.
        models (dict): Dictionary containing the models to use for inference.
        transformations (dict): Dictionary containing the transformations to apply to each model's inputs.
        input_dir (str): Path to the directory containing the video files.

    Returns:
        dict: Dictionary containing the output of each model's inference for each video file.
    """
    # Get a list of all video files in the input directory
    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mp4')]  # adjust the condition based on your video file format

    output_dict = {}
    for task_type, task_models in models.items():
        output_dict[task_type] = {}
        for model_name, model in task_models.items():
            framework = config.tasks[task_type][model_name].inference.framework
            if framework == 'torch':
                output_dict[task_type][model_name] = torch_inference.infer_videos(video_files, model, transformations[task_type][model_name], config, task_type, model_name)
    return output_dict
from scripts import torch_inference
from scripts import torchhub_inference
from scripts import pytorchvideo_inference
from scripts import huggingface_inference
import os

def inference(config, models, transformations, clip_durations, classes, input_dir):
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
            framework = config.tasks[task_type][model_name].load_model.framework
            if framework == 'torch':
                output_dict[task_type][model_name] = torch_inference.infer_videos(video_files, model, transformations[task_type][model_name], config, task_type, model_name)
            elif framework == 'torchhub':
                output_dict[task_type][model_name] = torchhub_inference.infer_videos_torchhub(video_files, model, transformations[task_type][model_name], clip_durations[task_type][model_name], classes[task_type][model_name], model_name)
            elif framework == 'pytorchvideo':
                if task_type == 'action_detection':
                   # Check if there is any output under the key "object_detection"
                    if "object_detection" in output_dict and any(output_dict["object_detection"].values()):
                        # If there is, proceed with the inference
                        
                        object_detection_output = next(iter(output_dict["object_detection"].values()))  # get the output of any model
                        output_dict[task_type][model_name] = pytorchvideo_inference.infer_videos(video_files, model, transformations[task_type][model_name], clip_durations[task_type][model_name], classes[task_type][model_name], model_name, object_detection_output)
                    else:
                        print("No output from object_detection. Cannot proceed with action_detection.") 
                else:
                    output_dict[task_type][model_name] = pytorchvideo_inference.infer_videos(video_files, model, transformations[task_type][model_name], clip_durations[task_type][model_name], classes[task_type][model_name], model_name)
            elif framework == 'huggingface':
                output_dict[task_type][model_name] = huggingface_inference.infer_videos_huggingface(video_files, model, transformations[task_type][model_name], model_name)
    return output_dict
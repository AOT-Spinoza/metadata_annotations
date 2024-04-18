from scripts import torch_inference
from scripts import torchhub_inference
from scripts import pytorchvideo_inference
from scripts import huggingface_inference
import os
import random

import os

import os

import os

def get_dirs_with_subdir(dirs, subdir_name):
    """
    Returns a dictionary where the keys are directory names and the values are True if the directory contains a subdirectory with the given name, and False otherwise.

    Args:
        dirs (list): List of directory paths.
        subdir_name (str): Name of the subdirectory to check for.

    Returns:
        dict: Dictionary where the keys are directory names and the values are True if the directory contains a subdirectory with the given name, and False otherwise.
    """
    dirs_with_subdir = {}
    for d in dirs:
        subdirs = os.listdir(d)
        dirs_with_subdir[d] = subdir_name in subdirs
    true_dirs = [keys for keys, values in dirs_with_subdir.items() if values]
    return true_dirs

def check_missing_counterparts(video_files, output_dir, n):
    """
    Checks for each video if its reversed counterpart exists and removes them from the list.

    Args:
        video_files (list): List of video file paths.
        output_dir (str): Path to the output directory.
        n (int): Number of video files to return.

    Returns:
        list: List of video file paths that have a reversed counterpart.
    """
    video_files_basename = [os.path.basename(f) for f in video_files]
    missing_counterparts = []

    output_dir = [os.path.basename(f) for f in output_dir]

    for video in output_dir:
        if len(missing_counterparts) >= n:
            break

        if video.startswith('S_'):

            reversed_video = 'R_' + video
        elif video.startswith('R_S_'):
 
            reversed_video = video[2:]
        else:
            continue

        if reversed_video not in output_dir:
            missing_counterparts.append(video)

    # Find the full paths of the missing counterparts in the video_files list
    missing_counterparts_paths = [f for f in video_files if os.path.basename(f) in missing_counterparts]

    return missing_counterparts_paths

def get_unused_video_files(video_files, output_dir, number_video):
    """
    Filters the video files to include only those that are not already names of directories under the output directory.

    Args:
        video_files (list): List of video file paths.
        output_dir (str): Path to the output directory.

    Returns:
        list: List of video file paths that are not already names of directories under the output directory.
    """
    video_files.sort()
    selected_video_files = video_files
    existing_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    unused_video_files = [f for f in selected_video_files if os.path.basename(f) not in existing_dirs]
    
    if len(unused_video_files) < number_video:
        print(f'Found {len(unused_video_files)} unused video files, which is less than the requested {number_video} video files.')
    else:
        print(f'Found {len(unused_video_files)} unused video files. Returning list of {number_video} video files.')
    return unused_video_files

def get_video_dirs_without_subdir(video_dirs, subdir, number_video):
    """
    Filters the video directories to include only those that do not contain the specified subdirectory.

    Args:
        video_dirs (list): List of video directory paths.
        subdir (str): Name of the subdirectory to check for.
        number_video (int): Number of video directories to select.

    Returns:
        list: List of video directory paths that do not contain the specified subdirectory.
    """
    video_dirs_without_subdir = [d for d in video_dirs if subdir not in os.listdir(d)]
    
    if number_video > len(video_dirs_without_subdir):
        print(f"Requested {number_video} videos, but only found {len(video_dirs_without_subdir)}. Returning all available videos.")
        return video_dirs_without_subdir

    selected_video_dirs = random.sample(video_dirs_without_subdir, number_video)
    return selected_video_dirs


def get_video_dirs_without_files(video_dirs, subdirs, file_extension):
    """
    Filters the video directories to include only those that do not contain any files with the specified extension
    in any of the specified subdirectories.

    Args:
        video_dirs (list): List of video directory paths.
        subdirs (list): List of subdirectory names to check for.
        file_extension (str): File extension to check for.

    Returns:
        list: List of video directory paths that do not contain any files with the specified extension in any of the specified subdirectories.
    """
    video_dirs_without_files = []

    for video_dir in video_dirs:
        for subdir in subdirs:
            subdir_path = os.path.join(video_dir, subdir)
            if os.path.isdir(subdir_path):
                for root, dirs, files in os.walk(subdir_path):
                    if not any(file.endswith(file_extension) for file in files):
                        video_dirs_without_files.append(video_dir)
                        break

    return video_dirs_without_files

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
    ## Get a list of all video files in the input directory
    video_files =input_dir
    
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
                        object_detection_output = next(iter(output_dict["object_detection"].values()))  # get the output of any model
                        output_dict[task_type][model_name] = pytorchvideo_inference.infer_videos(config, video_files, model, transformations[task_type][model_name], clip_durations[task_type][model_name], classes[task_type][model_name], model_name, object_detection_output)
                    else:
                        ## TO DO load in the object_detection info from results if it is exists and use that as input to the action detection
                        print("No output from object_detection. Cannot proceed with action_detection.") 
                else:
                    output_dict[task_type][model_name] = pytorchvideo_inference.infer_videos(config, video_files, model, transformations[task_type][model_name], clip_durations[task_type][model_name], classes[task_type][model_name], model_name)
            elif framework == 'huggingface':
                output_dict[task_type][model_name] = huggingface_inference.infer_videos_huggingface(video_files, model, transformations[task_type][model_name], model_name)
    return output_dict
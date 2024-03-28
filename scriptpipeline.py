from src.load_models_and_config import load_all_models_and_metadata
from src.predictions import inference
from src.postprocessing import postprocess_predictions
import src.transformations as transformations
from src.save_data import determine_and_execute_export_function
from scripts.statistical_analysis import execute_all_statistical_analysis
import hydra_zen
import numpy as np
import os
import gc
import torch
from tqdm import tqdm
from  src.predictions import get_dirs_with_subdir, get_video_dirs_without_subdir, get_unused_video_files, check_missing_counterparts
config = hydra_zen.load_from_yaml("config.yaml")


def input_function(config,input_dir):

    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mp4')]  # adjust the condition based on your video file format
    
    # FILTERING OPTIONS now you have to comment out which you want to do but will become part of confg later on, also you can give the number of videos you want to process
    # OPTION 1 Filter the selected video files from INPUT to include only those that are not already names of directories under the OUTPUT directory, so that we don't process the same video twice
    # video_files = get_unused_video_files(video_files, "/tank/shared/2024/visual/AOT/derivatives/stimuli/annotations",100)
    # # OPTION 2 Check for each video in OUTPUT dir if its reversed time counterpart exists and use the one that is missing
    # # Get a list of all video directories in the output directory
    #  output_dir = config['outputs']
    #  video_dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    #  video_files = check_missing_counterparts(video_files, video_dirs, 150)
    # # OPTION 3 Filter the selected video directories in the OUTPUT dir to include only those that do not contain the a subdirectory in OUTPUT (task)
    #  output_dir = config['outputs']
    #  video_dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    #  video_dirs = get_video_dirs_without_subdir(video_dirs,'instance_segmentation', 2)
    #  video_files = [d.replace(output_dir, input_dir) for d in video_dirs]
# /tank/tgn252/metadata_annotations/result/R_S_yt-aiwDdJm1ek8_90.mp4/instance_segmentation/MaskRCNN_ResNet50_FPN/R_S_yt-aiwDdJm1ek8_90_MaskRCNN_ResNet50_FPN.mp4
    # Option 4 Filter the selected videos directories in OUTFUT that only include a specific subdir(task) in OUTPUT
    # output_dir = config['outputs']
    # video_dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    # video_files = get_dirs_with_subdir(video_dirs, 'instance_segmentation')
    # video_files = get_video_dirs_without_subdir(video_files, 'action_classification', 60)
    # video_files = [d.replace(output_dir, input_dir) for d in video_files]
    # print(video_files)
    
    return video_files

def my_pipeline(config):
    """
    Executes a pipeline of operations on the given configuration.

    Args:
        config (dict): The configuration for the pipeline.

    Returns:
        None
    """
    # Load the models and transformations.
    models, transformations, classes, clip_durations = load_all_models_and_metadata(config)
    inputs = input_function(config, config["inputs"])
    # if you do batches just do this:
    inputs = inputs[:100]

    print(len(inputs), "video files to process")
    for input in tqdm(inputs):
        print('processing', input)
        outputs = inference(config, models, transformations, clip_durations, classes, [input])
        postprocessed = postprocess_predictions(outputs, config)
        determine_and_execute_export_function(postprocessed,classes, config)
        torch.cuda.empty_cache()
        gc.collect()    
        print('out of determine_and_execute_export_function')
    # execute_all_statistical_analysis(postprocessed, config)
    
my_pipeline(config)

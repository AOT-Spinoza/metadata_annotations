
from torchvision.transforms.functional import convert_image_dtype
import numpy as np
import os
import tqdm
import pickle
import json
import torch
import sys
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from src.video_reader import custom_read_video
import torchvision.io
import sys


def infer_videos(model, transformation, video_files, model_config, model_name):
    """
    Runs inference on the given video files using the specified model and transformation.

    Args:
        model: The model to use for inference.
        transformation: The transformation to apply to each frame before inference.
        video_files (list): List of paths to the video files.
        model_config (DictConfig): The Hydra configuration for the model.

    Returns:
        dict: A dictionary with model names as keys, video names as sub-keys, and a list of predictions per frame as values.
    """
    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Move the model to the device
    model = model.to(device)
    # Initialize the output dictionary

    outputs_all = {model_name: {}}
    for video in video_files:
        video_reader = torchvision.io.VideoReader(video, "video")
        video_frames, _, pts, meta = custom_read_video(video_reader)
        # Initialize the list of predictions for this video
        outputs_all[model_name][os.path.basename(video)] = []
    
        # Retrieve preprocessing steps from config
        unsqueeze = model_config.preprocessing.unsqueeze
        to_tensor = model_config.preprocessing.to_tensor

        for frame_count, frame in enumerate(video_frames, start=1):
            # Apply preprocessing steps if they are specified
            if unsqueeze:
                frame = frame.unsqueeze(0)
            if to_tensor:
                frame = torch.from_numpy(frame)
            # Apply the transformation to the video frame.
            transformed_frame = transformation(frame)
            # Move the transformed frame to the device
            transformed_frame = transformed_frame.to(device)
            # Perform inference
            with torch.no_grad():
                output = model([transformed_frame])
            output = output.detach().cpu()
            # Store the output in the output list.
            outputs_all[model_config.name][os.path.basename(video)].append(output)
    return outputs_all
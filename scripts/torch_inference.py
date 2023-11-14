

from torchvision.transforms.functional import convert_image_dtype
# import cv2
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


## COPILOT VERSION
import sys
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from src.video_reader import custom_read_video
import torchvision.io

def infer_videos(inputs, model, transformation, model_info):
    """
    Performs inference on a list of videos using a given model and transformation.

    Args:
        inputs (list): A list of input video file paths.
        model: The model to use for inference.
        transformation: The transformation to apply to each frame before inference.
        model_info (dict): A dictionary containing the configuration for the model.

    Returns:
        list: A list of model outputs for each input video and frame.
    """
    
    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model = model.to(device)

    outputs_all = []
    for video in inputs:
        video_reader = torchvision.io.VideoReader(video, "video")
        video_frames, _, pts, meta = custom_read_video(video_reader)

        for frame_count, frame in enumerate(video_frames, start=1):
            # Apply preprocessing steps if they are specified in the model_info
            if 'unsqueeze' in model_info:
                frame = frame.unsqueeze(0)
            if 'to_tensor' in model_info:
                frame = torch.from_numpy(frame)

            # Apply the transformation to the video frame.
            transformed_frame = transformation(frame)

            # Move the transformed frame to the device
            transformed_frame = transformed_frame.to(device)

            # Perform inference
            with torch.no_grad():
                output = model([transformed_frame])

            # Detach the output from the computation graph and move it to CPU
            output = output.detach().cpu()

            # Store the output in the output list.
            outputs_all.append(output)

    return outputs_all

# def inference(inputs, model, transformation, config):
#     """
#     Runs inference on a set of input videos using a given model and transformation.

#     Args:
#         inputs (list): A list of input video file paths.
#         model: The model to use for inference.
#         transformation: The transformation to apply to each frame before inference.
#         config: Additional configuration options.

#     Returns:
#         dict: A dictionary containing the model outputs for each input video and frame.
#     """
    
#     outputs_all = {}
#     for video in inputs:
#         stream = "video"
#         video_reader = torchvision.io.VideoReader(video, stream)
#         video_frames, _, pts, meta = custom_read_video(video_reader)

#         for frame_count, frame in enumerate(video_frames, start=1):
#             # Apply the transformation to the video frame.
#             transformed_frame = transformation(frame)

#             # Convert the transformed frame to a PyTorch tensor, if necessary.
#             if config.model_name.get("to_tensor"):
#                 transformed_frame = torch.from_numpy(transformed_frame)

#             # Unsqueeze the input tensor, if necessary.
#             if config.model_name.get("unsqueeze"):
#                 transformed_frame = transformed_frame.unsqueeze(0)

#             # Pass the transformed frame to the model.
#             model.to_cuda()
#             output = model([transformed_frame])

#             # Store the output in the output dictionary.
#             outputs_all[video][frame_count] = output

#     return outputs_all

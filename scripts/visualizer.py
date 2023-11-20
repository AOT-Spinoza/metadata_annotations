
from torchvision.transforms.functional import to_pil_image, resize
from torchvision.utils import draw_segmentation_masks
from PIL import Image
from scripts.find_video import find_video_path
import torchvision
from src.video_reader import custom_read_video
import gc

import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = 'tight'


def export_frame(imgs, filename):
    """
    Export single frame or list and save the figure to a file.

    Args:
        imgs (list or torch.Tensor): List of images or a single image tensor.
        filename (str): The filename to save the figure.

    Returns:
        None
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(filename)

def visualize_segmentation(predictions, config):
    """
    Visualizes the segmentation predictions for different task types, models, and videos.

    Args:
        predictions (dict): A dictionary containing the segmentation predictions.
            The keys are task types, and the values are dictionaries containing
            the model names and their corresponding video predictions.
        config (dict): A dictionary containing the configuration settings.

    Returns:
        dict: A dictionary named video_masked containing the structure task_type{model{video{[masked_frames,metadata]}}}.

    """
    video_masked = {}
    # Iterate over different task types and their results
    for task_type, task_results in predictions.items():
        video_masked[task_type] = {}
        # Iterate over different models and their results
        for model_name, videos in task_results.items():
            video_masked[task_type][model_name] = {}
            # Iterate over different videos and their predictions
            for video_name, prediction in videos.items():
                video_masked[task_type][model_name][video_name] = []
                # Find the video path using the video name and config
                video = find_video_path(video_name, config)
                # Create a VideoReader for the video
                video_reader = torchvision.io.VideoReader(video, "video")
                # Read the video frames, audio, pts, and metadata
                video_frames, _, pts, meta = custom_read_video(video_reader)
                # Delete the VideoReader to free up memory
                del video_reader
                # Collect any garbage left from deleting the VideoReader
                gc.collect()
                # Resize each frame to a height of 520 while maintaining the aspect ratio
                resized_frames = [resize(frame, size=520) for frame in video_frames]
                # The dimension along which the classes are represented
                class_dim = 1
                # Stack the list of predictions into a single tensor
                prediction = torch.stack(prediction)
                # Get the number of classes from the shape of the prediction tensor
                num_classes = prediction.shape[class_dim]
                # Create a binary mask for each class
                all_classes_masks = (prediction.argmax(class_dim) == torch.arange(num_classes)[:, None, None, None])
                # The shape of all_classes_masks is (C, T, H, W) so we need to swap first two axes
                all_classes_masks = all_classes_masks.swapaxes(0, 1)
                masked_frames = [draw_segmentation_masks(frame, masks=mask, alpha=0.5) for frame, mask in zip(resized_frames, all_classes_masks)]
                video_masked[task_type][model_name][video_name] = [masked_frames, meta]
    return video_masked

def create_videos_from_frames(data, video_name, task_type, config):
    """
    Create a video file from a list of frames.

    Args:
        frames (list): List of frames to be included in the video.
        video_name (str): Name of the output video file.
        metadata (dict): Metadata containing information about the video.

    Returns:
        None
    """
    # Create a VideoWriter object   
    # Convert the list of frames to a 4D tensor
    frames_tensor , 
    frames_tensor = torch.stack(frames)
    print(frames_tensor.shape)
    frames_tensor = frames_tensor.permute(0, 2, 3, 1)
    print(frames_tensor.shape) 
    # Write the frames to a video file
    print(metadata['video']['fps'])
    torchvision.io.write_video(video_name, frames_tensor, metadata['video']['fps'][0])


import os
from torchvision.transforms.functional import resize
from torchvision.utils import draw_segmentation_masks
from PIL import Image
from scripts.find_video import find_video_path
import torchvision
from src.video_reader import custom_read_video
import gc
import random
import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
import torch.nn.functional as FF

plt.rcParams["savefig.bbox"] = 'tight'

def get_video_data(video_name, config, resize_value=None):
    """
    Retrieves video frames and frames per second (fps) from a given video file.

    Args:
        video_name (str): The name of the video file.
        config (dict): Configuration settings.
        resize_value (tuple, optional): The desired size to resize the frames. Defaults to None.

    Returns:
        tuple: A tuple containing the resized frames and the frames per second (fps).
    """


    video_name += ".mp4"
    video = find_video_path(video_name, config)
    video_reader = torchvision.io.VideoReader(video, "video")
    video_frames, _, pts, meta = custom_read_video(video_reader)
    fps = meta['video']['fps'][0]
    del video_reader
    gc.collect()

    if resize_value is not None:
        resized_frames = [resize(frame, size=resize_value) for frame in video_frames]
    else:
        resized_frames = video_frames
    return resized_frames, fps


def write_video(out_video_name, frames, fps):
    """
    Write a video file from a list of frames. The frames are expected to be in the format [T, C, H, W].
    Will be permuted into [T, H, W, C] for the video writer that is expected.

    ## TO DO need to think if permuting must be into config if other frame formats are used for upcoming models.
    

    Args:
        out_video_name (str): The name of the output video file.
        frames (list): A list of frames to be written to the video file.
        fps (int): The frames per second of the output video.

    Returns:
        None
    """
    
    frames_tensor = torch.stack(frames)
    frames_tensor = frames_tensor.permute(0, 2, 3, 1)
    torchvision.io.write_video(out_video_name, frames_tensor, fps)


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


def create_semantic_masks(predictions, video_name, config, resize_value=None):
    """
    Visualizes the segmentation predictions for a single video.

    Args:
        predictions (list): A list containing the segmentation predictions for the video.
        video_name (str): The name of the video.
        config (dict): A dictionary containing the configuration settings.

    Returns:
        list: A list containing the masked frames and metadata for the video.
    """
    
    # Find the video path using the video name and config
    frames, fps = get_video_data(video_name, config, resize_value)                        
    # The dimension along which the classes are represented
    class_dim = 1
    # Stack the list of predictions into a single tensor
    prediction = torch.stack(predictions)
    # Get the number of classes from the shape of the prediction tensor
    num_classes = prediction.shape[class_dim]
    # Create a binary mask for each class
    all_classes_masks = (prediction.argmax(class_dim) == torch.arange(num_classes)[:, None, None, None])
    # The shape of all_classes_masks is (C, T, H, W) so we need to swap first two axes
    all_classes_masks = all_classes_masks.swapaxes(0, 1)

    # Apply the masks to the frames
    masked_frames = [draw_segmentation_masks(frame, masks=mask, alpha=0.5) for frame, mask in zip(frames, all_classes_masks)]

    return masked_frames, fps



def create_videos_from_frames(data, out_video_name, task_type,video_name, config, resize_value, classes=None):
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
    if task_type == 'depth_estimation':
        _, fps = get_video_data(video_name, config, resize_value)
        # Normalize the output to the range 0-255
        frames = []
        for output in data:
            output = (output - output.min()) / (output.max() - output.min()) * 255
            output = output.type(torch.uint8)
            # The output of MiDaS is a single-channel depth map, so we repeat it to create a 3-channel image
            output = output.squeeze(0).repeat(3, 1, 1)
            output = output.squeeze(1)
            frames.append(output)
        write_video(out_video_name, frames, fps)

    if task_type == 'instance_segmentation':
        resized_frames, fps = get_video_data(video_name, config, resize_value)
        score_threshold = .75
        prob_threshold = .5
        boolean_masks = [
            (out['masks'][out['scores'] > score_threshold] > prob_threshold) for out in data]

        frames_with_masks = [
            draw_segmentation_masks(img, mask.squeeze(1))
            for img, mask in zip(resized_frames, boolean_masks)
        ]
        write_video(out_video_name, frames_with_masks, fps)

    if task_type == 'semantic_segmentation':
        video_masked, fps = create_semantic_masks(data,video_name, config, resize_value)	
        write_video(out_video_name, video_masked, fps)
    if task_type == 'keypoints':
        # Create a dictionary to store the colors for each unique ID
        id_to_color = {}
        video_frames, fps = get_video_data(video_name, config, resize_value)
        overdrawn_frames = []
        for frame, prediction in zip(video_frames, data):
            keypoints = prediction['keypoints']
            ids = prediction['ids']
            for i, id in enumerate(ids):
                # Convert keypoints to the expected format [num_instances, K, 2]
                # Check if the unique ID is in the dictionary
                if id not in id_to_color:
                    # If it's not, generate a new color and add it to the dictionary
                    id_to_color[id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                # Get the color for the unique ID
                color = id_to_color[id]
                # Draw keypoints on the frame
                keypoints_xy = keypoints[i][:, :2]  # Take only the first two columns (x, y)
                keypoints_xy = keypoints_xy.unsqueeze(0)  # Add an extra dimension at the beginning
                
                ## frame will be a tensor of shape [C, H, W]
                frame = torchvision.utils.draw_keypoints(frame, keypoints_xy, colors=color)
            
            overdrawn_frames.append(frame)
        write_video(out_video_name, overdrawn_frames, fps)
    if task_type == 'object_detection':
        video_frames, fps = get_video_data(video_name, config, resize_value)
        overdrawn_frames = []
        for frame, prediction in zip(video_frames, data):

            # Get the bounding boxes and labels from the prediction
            boxes = prediction['boxes']
            labels = [classes[i] for i in prediction['labels']]  
            # Draw the bounding boxes on the frame
            frame = torchvision.utils.draw_bounding_boxes(frame, boxes, labels, font="/tank/tgn252/metadata_annotations/library/GothamMedium.ttf", font_size=20, width=4)
            overdrawn_frames.append(frame)
        write_video(out_video_name, overdrawn_frames, fps)

    if task_type == 'action_detection':
        # Create a dictionary to store the colors for each unique ID
        id_to_color = {}
        video_frames, fps = get_video_data(video_name, config, resize_value)
        overdrawn_frames = []
        for frame_number, (frame, prediction) in enumerate(zip(video_frames, data)):
            if prediction is None:  # Skip the frame if the prediction is None
                print(f"Empty prediction for frame number {frame_number}")
                continue
            
            # Get the bounding boxes, labels, and IDs from the prediction
            boxes = prediction['boxes']
            labels =  prediction['max_classes']
            ids = prediction['ids']
            # Draw the bounding boxes on the frame with colors based on IDs
            for i, id in enumerate(ids):
                # Check if the unique ID is in the dictionary
                if id not in id_to_color:
                    # If it's not, generate a new color and add it to the dictionary
                    id_to_color[id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                # Get the color for the unique ID
                color = id_to_color[id]
                # Draw the bounding box with the color
                frame = torchvision.utils.draw_bounding_boxes(frame, boxes[i].unsqueeze(0), [labels[i]], colors=color, font="/tank/tgn252/metadata_annotations/library/GothamMedium.ttf", font_size=20, width=4)

            overdrawn_frames.append(frame)
        write_video(out_video_name, overdrawn_frames, fps)
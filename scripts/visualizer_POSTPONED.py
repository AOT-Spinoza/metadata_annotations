import numpy as np
import cv2
import torch
import os
from PIL import Image
from src.video_reader import custom_read_video
import torchvision
from torchvision.transforms import functional as F

class Visualizer:
    def __init__(self, class_names, class_colors):
        self.class_names = class_names
        self.class_colors = class_colors
    
    def resize_frame(self, frame, size):
        # Convert the frame to a PIL Image
        frame = Image.fromarray(frame)
        # Resize the frame
        frame = frame.resize(size, Image.BILINEAR)
        # Convert the frame back to a numpy array
        frame = np.array(frame)
        return frame

    def draw_sem_seg(self, frame, masks, alpha=0.5):
        # Resize the frame to match the size of the masks
        frame = self.resize_frame(frame, (924, 520))  # or (520, 924) depending on the orientation
        # Convert the PIL Image to a numpy array
        frame = np.array(frame)
        # Create an empty image to store the visualizations
        vis_image = np.zeros(frame.shape, dtype=np.uint8)
        for i, mask in enumerate(masks):
            # Choose a color for this class
            color = self.class_colors[i % len(self.class_colors)]
            # Draw the mask on the image
            vis_image[mask] = color
        # Blend the visualizations with the original image
        frame = cv2.addWeighted(frame, 1 - alpha, vis_image, alpha, 0)
        return frame

def visualize(predictions, config):
    """
    Visualizes the predictions made by the models.

    Args:
        predictions (dict): A dictionary containing the predictions made by the models.
        config (dict): The configuration dictionary.
    """
    # Define the class names and colors
    class_names = [
        "Person",
        "Bird",
        "Cat",
        "Cow",
        "Dog",
        "Horse",
        "Sheep",
        "Aeroplane",
        "Bicycle",
        "Boat",
        "Bus",
        "Car",
        "Motorbike",
        "Train",
        "Bottle",
        "Chair",
        "Dining Table",
        "Potted Plant",
        "Sofa",
        "TV/Monitor"
    ]

    class_colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(len(class_names))]

    # Initialize the visualizer
    v = Visualizer(class_names, class_colors)

    # Visualize the predictions
    visualized = {}
    for task_type, task_results in predictions.items():
        visualized[task_type] = {}
        for model_name, videos in task_results.items():
            visualized[task_type][model_name] = {}
            for video_name, prediction in videos.items():
                # Construct the video file path
                video_path = os.path.join(config["inputs"], video_name)
                # Create a VideoReader object
                video_object = torchvision.io.VideoReader(video_path, "video")
                # Load the video into frames
                video_frames, _, _, _ = custom_read_video(video_object, read_video=True, read_audio=False)
                # Permute the frames to the shape (T, H, W, C)
                video_frames = video_frames.permute(0, 2, 3, 1).numpy()
                # Resize the frames to match the masks
                video_frames = [v.resize_frame(frame, (520, 924)) for frame in video_frames]
                # Convert the prediction to boolean masks
                class_dim = 1
                for i, single_prediction in enumerate(prediction):
                    num_classes = single_prediction.shape[class_dim]
                    all_classes_masks = single_prediction.argmax(class_dim) == torch.arange(num_classes)[:, None, None, None]
                    all_classes_masks = all_classes_masks.swapaxes(0, 1)
                    all_classes_masks = [F.resize(mask, frame.size[::-1]) for mask, frame in zip(all_classes_masks, video_frames)]
                    # Draw the segmentation masks on the images
                    visualized_prediction = [
                        v.draw_sem_seg(frame, masks=mask, alpha=0.6)
                        for frame, mask in zip(video_frames, all_classes_masks)
                    ]
                    visualized[task_type][model_name][video_name] = visualized_prediction
    return visualized
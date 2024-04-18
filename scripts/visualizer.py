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
from matplotlib import colors as mcolors
import torchvision.transforms.functional as F
import torch.nn.functional as FF
from PIL import ImageColor
from torchvision.ops import box_convert
plt.rcParams["savefig.bbox"] = 'tight'


import subprocess


import os
import subprocess
from torchvision.utils import save_image
import subprocess
import torch

def write_video_directly(out_video_name, frames, fps):
    height, width = frames[0].shape[1], frames[0].shape[2]
    command = [
        'ffmpeg',
        '-y',  # overwrite output file if it exists
        '-f', 'rawvideo',  # input format
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',  # size of one frame
        '-pix_fmt', 'rgb24',  # format, using rgb24 as our input is in RGB
        '-r', str(fps),  # frames per second
        '-i', '-',  # The input comes from a pipe
        '-an',  # Tells FFmpeg not to expect any audio
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-crf', '23',
        out_video_name
    ]

    # Open a pipe to FFmpeg
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    # Write frames to pipe
    for frame in frames:
        # Convert PyTorch tensor to numpy array and transpose it to HWC format
        frame_np = frame.cpu().numpy().transpose(1, 2, 0)
        # Ensure the data type is uint8
        # frame_np = (frame_np * 255).astype('uint8')
        # Write frame data
        process.stdin.write(frame_np.tobytes())

    # Close the pipe and wait for FFmpeg to finish
    process.stdin.close()
    process.wait()

    if process.returncode != 0:
        raise RuntimeError('FFmpeg returned error')





# def write_images_and_video(out_video_name, frames, fps):
#     # Get the directory name from out_video_name
#     out_dir = os.path.dirname(out_video_name)

#     # Save each frame as an image
#     image_paths = []
#     for i, frame in enumerate(frames):
#         # Normalize the frame to [0, 1]
#         frame = frame.float() / 255
#         image_path = os.path.join(out_dir, f'frame_{i:04d}.png')
#         save_image(frame, image_path)
#         image_paths.append(image_path)

#     # Use FFmpeg to convert the images into a video
#     subprocess.run(['ffmpeg', '-y', '-framerate', str(fps), '-i', os.path.join(out_dir, 'frame_%04d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', out_video_name])

#     # Remove the images
#     for image_path in image_paths:
#         os.remove(image_path)

def get_video_data(video_name, config, resize_value=None):
    """
    Retrieves video frames and frames per second (fps) from a given video file.

    Args:
        video_name (str): The name of the video file.
        config (dict): Configuration settings.
        resize_value (tuple, optional): The desired size to resize the frames. Defaults to None.

    Returns:
        tuple: A tuple containing the resized frames and the frames per second (fps).
           The video_frames tensor has the following dimensions: [T,C,H,W]
    - T: number of frames in the video
    - C: number of channels in the video (3 for RGB)
    - H: height of each frame in pixels
    - W: width of each frame in pixels
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


from matplotlib import colors as mcolors

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

    # Generate a consistent color map for all possible classes
    color_map = mcolors.CSS4_COLORS
    color_map = list(color_map.values())  # Convert to list to allow indexing
    # Convert the CSS4 colors to RGB tuples and then to tensors
    

    # Set the color for class 0 to black (optional)
    color_map[0] = torch.tensor((0, 0, 0))
    # Apply the masks to the frames
    masked_frames = [draw_segmentation_masks(frame, masks=mask, alpha=0.5, colors=color_map) for frame, mask in zip(frames, all_classes_masks)]

    return masked_frames, fps



def create_videos_from_frames(data, out_video_name, task_type,video_name, config, resize_value=None, classes=None):
    """
    Create a video file from a list of frames.

    Args:
        frames (list): List of frames to be included in the video.
        video_name (str): Name of the output video file.
        metadata (dict): Metadata containing information about the video.

    Returns:
        None
    """

    print(out_video_name)

    # Create a VideoWriter object   
    if task_type == 'depth_estimation':
        original_frames, fps = get_video_data(video_name, config, resize_value)
        original_size = original_frames[0].shape[-2:]
        # Normalize the output to the range 0-255
        frames = []
        for output in data:
            output = output.unsqueeze(1)
            output = torch.nn.functional.interpolate(
                        output,
                        size=original_size,
                        mode="bicubic",
                        align_corners=False,)
            
            output = (output - output.min()) / (output.max() - output.min()) * 255
            output = output.type(torch.uint8)
            # The output of MiDaS is a single-channel depth map, so we repeat it to create a 3-channel image
            output = output.squeeze(0).repeat(3, 1, 1)
            output = output.squeeze(1)
            frames.append(output)
        write_video_directly(out_video_name, frames, fps)




    if task_type == 'instance_segmentation':
        height, width = None, None
        model_name = "MaskRCNN_ResNet50_FPN"
        
        for frame in data:
            try:
                height, width = frame['masks'].shape[-2:]
                break
            except:
                continue
        if height is None or width is None:
            # Get the directory and base name of out_video_name
            dir_name = os.path.dirname(out_video_name)
            base_name = os.path.basename(out_video_name)
            # Remove the extension from the base name
            base_name_without_ext = os.path.splitext(base_name)[0]

            # Create the text file name
            txt_file_name = f"{base_name_without_ext}.txt"

            # Create the full path to the text file
            txt_file_path = os.path.join(dir_name, txt_file_name)

            with open(txt_file_path, 'w') as f:
                f.write("No instances detected in video")
            print(f"No instances, text fle saved to {txt_file_path}")
            return None
        resized_frames, fps = get_video_data(video_name, config, height)
        score_threshold = .75
        prob_threshold = .5
        # Create a colormap
        cmap = plt.get_cmap('tab20')

        # Create a dictionary to store the colors for each ID
        id_to_color = {}

        boolean_masks = []
        color_lists = []
        
        for out in data:
            
            mask = (out['masks'][out['scores'] > score_threshold] > prob_threshold) if out['masks'].numel() > 0 else torch.zeros((1, 1, height, width), dtype=torch.bool)
            boolean_masks.append(mask)

            # Get the IDs for the current frame
            frame_ids = out['ids']
            if len(frame_ids) == 0:
                color_lists.append([])
                continue
            
            
            # Create a color list for the current frame
            color_list = []
            for id in frame_ids:
                # Convert tensor to int
                id_int = id.item()

                if id_int not in id_to_color:
                    # Assign a new color to the ID
                    id_to_color[id_int] = cmap(len(id_to_color) % 20)
                color_list.append(id_to_color[id_int])

            # Convert the colors to RGB tuples
            color_list = [(int(color[0]*255), int(color[1]*255), int(color[2]*255)) for color in color_list]
            color_lists.append(color_list)

        frames_with_masks = []
        for i, (img, mask, color_list) in enumerate(zip(resized_frames, boolean_masks, color_lists)):
            if len(color_list) > 0:
                try:
                    frame_with_mask = draw_segmentation_masks(img, mask.squeeze(1), colors=color_list)
                    frames_with_masks.append(frame_with_mask)
                except ValueError as e:
                    print(f"Error occurred at frame {i}: {e}")
                    continue
            else:
                frames_with_masks.append(img)

        write_video_directly(out_video_name, frames_with_masks, fps)
        return 1
    if task_type == 'semantic_segmentation':
        video_masked, fps = create_semantic_masks(data,video_name, config, resize_value)	
        write_video_directly(out_video_name, video_masked, fps)
    if task_type == 'keypoints':
        # Create a dictionary to store the colors for each unique ID
        id_to_color = {}
        video_frames, fps = get_video_data(video_name, config, resize_value)
        overdrawn_frames = []

        for frame, prediction in zip(video_frames, data):

            keypoints = prediction['keypoints']
            ids = prediction['ids']

            for i, id in enumerate(ids):
                if i < len(keypoints):
                    id = int(id)
                # Convert keypoints to the expected format [num_instances, K, 2]
                    # Check if the unique ID is in the dictionary
                    if id not in id_to_color.keys():
                        # If it's not, generate a new color and add it to the dictionary
                        id_to_color[id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    # Get the color for the unique ID
                    color = id_to_color[id]
                    # Draw keypoints on the frame
                    keypoints_xy = keypoints[i][:, :2]  # Take only the first two columns (x, y)
                    keypoints_xy = np.expand_dims(keypoints_xy, axis=0)  # Add an extra dimension at the beginning
                    keypoints_xy = torch.from_numpy(keypoints_xy)  # Convert to a tensor
                    ## frame will be a tensor of shape [C, H, W]
                    frame = torchvision.utils.draw_keypoints(frame, keypoints_xy, colors=color)
            overdrawn_frames.append(frame)
        write_video_directly(out_video_name, overdrawn_frames, fps)


    if task_type == 'object_detection':
        video_frames, fps = get_video_data(video_name, config, resize_value)
        overdrawn_frames = []
        for frame, prediction in zip(video_frames, data):

            # Get the bounding boxes and labels from the prediction
            boxes = prediction['boxes']
            print(boxes)
            labels = [classes[i] for i in prediction['labels']]  

            # Ensure boxes are float32 and on the same device as the frame
            boxes = boxes.to(frame.device).float()

            # Check that xmin < xmax and ymin < ymax for all boxes
            # Skip if boxes is empty
            if boxes.nelement() != 0:
                # Ensure boxes is a 2D tensor
                if boxes.dim() == 1:
                    boxes = boxes.unsqueeze(0)

                # Create a mask for boxes that meet the condition
                mask = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])

                # Apply the mask to boxes and labels
                boxes = boxes[mask]
                labels = [label for box, label in zip(mask, labels) if box]

            # Draw the bounding boxes on the frame
            if len(boxes) > 0:
                boxes = torch.where(boxes < 0, torch.zeros_like(boxes), boxes)
                frame = torchvision.utils.draw_bounding_boxes(frame, boxes, labels, font="/tank/tgn252/metadata_annotations/library/GothamMedium.ttf", font_size=20, width=4)         
            overdrawn_frames.append(frame)
        write_video_directly(out_video_name, overdrawn_frames, fps)

    if task_type == 'action_detection':
        # Create a dictionary to store the colors for each unique ID
        id_to_color = {}
        video_frames, fps = get_video_data(video_name, config, resize_value)
        overdrawn_frames = []
        for frame_number, (frame, prediction) in enumerate(zip(video_frames, data)):
            if prediction is None:  # Skip the frame if the prediction is None
                print(f"Empty prediction for frame number {frame_number}")
                overdrawn_frames.append(frame)
                continue
            

            # Get the bounding boxes, labels, and IDs from the prediction
            boxes = prediction['boxes']
            if len(boxes) == 0:
                overdrawn_frames.append(frame)
                continue
            labels =  prediction['max_classes']
            ids = prediction['ids']

            # Prepare a list for colors
            colors_list = []

            for id in ids:
                id = int(id.item())  # Convert tensor to int
                # Check if the unique ID is in the dictionary
                if id not in id_to_color:
                    # If it's not, generate a new color and add it to the dictionary
                    id_to_color[id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                # Get the color for the unique ID
                color = id_to_color[id]
                # Add the color to the list
                colors_list.append(color)
            
            # Draw all the bounding boxes at once
            frame = torchvision.utils.draw_bounding_boxes(frame, boxes, labels=labels, colors=colors_list, font="/tank/tgn252/metadata_annotations/library/GothamMedium.ttf", font_size=20, width=4)
            overdrawn_frames.append(frame)
        write_video_directly(out_video_name, overdrawn_frames, fps)
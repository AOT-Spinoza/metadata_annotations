from pytorchvideo.data.encoded_video import EncodedVideo
from tqdm import tqdm
import torch
import numpy as np
import os
from src.save_data import load_data_from_hdf5
from src.tracker import tracking
from src.postprocessing import threshold
import os

def get_person_bboxes(video_name, output_object_detection=None):
    """
    Get bounding boxes of detected persons in a video.

    Args:
        video_name (str): The name of the video.
        output_object_detection (dict): Optional; The output from object detection.

    Returns:
        list: A list of dictionaries representing the bounding boxes of detected persons.
            Each dictionary contains the following keys:
            - 'frame': The frame number.
            - 'labels': The labels associated with the frame.
            - 'bboxes': The bounding boxes of the detected persons in the frame.

    Raises:
        FileNotFoundError: If the HDF5 file containing the detection results does not exist.
    """
    if output_object_detection is not None:
        print("Using output from object detection")
        # Use output_object_detection to get the bounding boxes
        # The exact code depends on the structure of output_object_detection
        detection_data = output_object_detection[os.path.basename(video_name)]
        print("detection_data correctly loaded")
    else:
        print("Searching for object detection output in results folder")
        # Extract the base video name and remove the file extension
        base_video_name = os.path.basename(video_name)
        base_video_name_without_extension = os.path.splitext(base_video_name)[0]
        
        # Define the task map and model name
        task_map = "object_detection"
        model_name = "fasterrcnn_resnet50_fpn_v2"

        # Construct the HDF5 file path
        map_name = f"{task_map}/{model_name}"
        hdf5_file_name = f"./result/{base_video_name}/{map_name}/{base_video_name_without_extension}_{model_name}.hdf5"

        # Check if the HDF5 file exists
        if not os.path.exists(hdf5_file_name):
            raise FileNotFoundError("No object detection found, make sure object detection is above the action detection in the config, or first do object detection on video")

        # Convert hdf5_file to a list of dictionaries using your custom function
        detection_data = load_data_from_hdf5(hdf5_file_name)

    filtered_prediction = threshold(detection_data, 0.8)

    detection_data = [
        {
            'boxes': frame['boxes'][frame['labels'] == 1],
            'labels': frame['labels'][frame['labels'] == 1],
            'scores': frame['scores'][frame['labels'] == 1],
        }
        for frame in filtered_prediction
    ]

    tracked_detection = tracking(filtered_prediction, None)
    detection_data = tracked_detection

    # Check if any frame has label 1 (person), if not no action recognition is needed
    has_label_1 = any((frame['labels'] == 1).any() for frame in detection_data)
    if not has_label_1:
        print("No person detected in video")
        return None

    return detection_data
    
    
def get_frame_boxes(detection_data, frame_number):
    """
    Get the bounding boxes for a specific frame number.

    Parameters:
    detection_data (dict): A dictionary containing detection data for multiple frames.
    frame_number (int): The frame number for which to retrieve the bounding boxes.

    Returns:
    person_boxes (list): A list of bounding boxes for people in the specified frame.
    """
    # Get the bounding boxes for the frame number
    frame_data = detection_data[frame_number]
    # Create a boolean mask where True corresponds to a label of 1
    mask = (frame_data['labels'] == 1)

    # Use the mask to select the boxes with a label of 1
    person_boxes = frame_data['boxes'][mask]
    person_ids = frame_data['ids'][mask]
    return person_boxes, person_ids


def infer_videos(video_files, model, transformation, clip_duration, classes, model_name, object_detection_output=None):
    """
    Infer actions in videos using a pre-trained model.

    Args:
        video_files (list): List of video file paths.
        model (torch.nn.Module): Pre-trained model for action recognition.
        transformation (callable): Transformation function to preprocess video clips and bounding boxes.
        clip_duration (float): Duration of each video clip in seconds.
        classes (list): List of class labels for action recognition.
        model_name (str): Name of the model.

    Returns:
        dict: Dictionary containing the inferred actions for each video and frame number.

    """

    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval()
    print(f"Using device: {device}")
    # Move the model to the device
    model = model.to(device)
    # Initialize the output dictionary
    outputs_all = {}
    
    for video_path in tqdm(video_files, desc=f"Processing videos for {model_name}"):
        # Get the bounding boxes of detected persons in the video
        detection_data = get_person_bboxes(video_path, object_detection_output)

        ## TODO: Think of output method for exporting videos with no persons
        if detection_data is None:
            print("Skipping video, no person detected")
            outputs_all[os.path.basename(video_path)] = {}
            continue

        # Load the video using EncodedVideo
        encoded_video = EncodedVideo.from_path(video_path)
        endbound = 2.6
        time_stamp_range = np.arange(clip_duration/2, endbound - clip_duration/2, clip_duration)
        frame_rate = 60
        total_frames_in_video = 150
        outputs_all[os.path.basename(video_path)] = [None] * total_frames_in_video

        for time_stamp in time_stamp_range:
            print("Generating predictions for time stamp: {} sec".format(time_stamp))
        
            # Generate clip around the designated time stamps
            inp_imgs = encoded_video.get_clip(
                time_stamp - clip_duration/2.0,  # start second
                time_stamp + clip_duration/2.0   # end second
            )
            inp_imgs = inp_imgs['video']
        
            # Calculate start_frame_number and middle_frame_number based on time_stamp and frame_rate
            start_frame_number = int((time_stamp - clip_duration/2.0) * frame_rate)
            middle_frame_number = int(time_stamp * frame_rate)
            end_frame_number = start_frame_number + inp_imgs.shape[1]

            # Generate people bbox predictions using Detectron2's off the shelf pre-trained predictor
            # We use the middle image in each clip to generate the bounding boxes.
            inp_img = inp_imgs[:, inp_imgs.shape[1]//2, :, :]
            inp_img = inp_img.permute(1, 2, 0)

            # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
            predicted_boxes, predicted_ids = get_frame_boxes(detection_data, middle_frame_number)
            if len(predicted_boxes) == 0:
                print("Skipping clip, no frames detected at time stamp: ", time_stamp)
                continue
            # Preprocess clip and bounding boxes for video action recognition.
            inputs, inp_boxes, _ = transformation(inp_imgs, predicted_boxes)
            # Prepend data sample id for each bounding box.
            # For more details, refer to the RoIAlign in Detectron2
            inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)

            # Generate action predictions for the bounding boxes in the clip.
            # The model here takes in the pre-processed video clip and the detected bounding boxes.
            if isinstance(inputs, list):
                inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
            else:
                inputs = inputs.unsqueeze(0).to(device)
            preds = model(inputs, inp_boxes.float().to(device))

            preds = preds.to('cpu')
            # The model is trained on AVA and AVA labels are 1 indexed, so prepend 0 to convert to 0 index.
            preds = torch.cat([torch.zeros(preds.shape[0], 1), preds], dim=1)
            # classes_dict = classes[0][0]
            # Get the class with the highest probability for each instance of a person
            max_probs, max_classes = torch.max(preds[:, 1:], dim=1)
            max_probs = max_probs.tolist()
            max_classes = max_classes.tolist()
            max_classes = [i+1 for i in max_classes]
            # Use the indices of the max probabilities to get the class names
            class_names = [classes[i] for i in max_classes]
            # Store the results in the output dictionary

            print(class_names)
            frame_output = {
                'boxes': predicted_boxes,
                'ids': predicted_ids,
                'max_classes': class_names,
                'max_probs': max_probs
            }

            # Assign the same output to all frames in the same clip as the middle frame
            clip_duration_frames = int(clip_duration * frame_rate)
            start_frame = max(0, middle_frame_number - clip_duration_frames // 2)
            end_frame = min(150, middle_frame_number + clip_duration_frames // 2)
            for frame_number in range(start_frame, end_frame):
                outputs_all[os.path.basename(video_path)][frame_number] = frame_output
                

    return outputs_all




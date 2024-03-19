from library.sort import Sort
import numpy as np
import torch
from library.deepsorttracker import DeepSortTracker
from scripts.visualizer import get_video_data
import cv2
import os
from torchvision.ops import box_iou


def get_frames_from_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}")

    frames = []
    while True:
        # Read each frame
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames left

        frames.append(frame)

    cap.release()
    return frames

def format_for_deepsort(detection, threshold=0.8):
    """
    Process the output from KeypointRCNN_ResNet50 for DeepSort.

    Args:
        keypoint_rcnn_output (list): Output from the KeypointRCNN_ResNet50 model.
        threshold (float): Confidence threshold to filter detections.

    Returns:
        Tuple of three lists: formatted bounding boxes (bbox_xywh), confidences, and class_ids.
    """
    formatted_bboxes = []
    confidences = []
    class_ids = [] 
     # Assuming you have class IDs, otherwise you can ignore this
    
    boxes = detection['boxes']  # Assuming boxes are in (x_min, y_min, x_max, y_max) format
    scores = detection['scores']
    labels = detection.get('labels', [0] * len(scores))  # Default to 0 if no labels
    # Filter out detections with low confidence
    indices = scores > threshold
    boxes = boxes[indices]
    scores = scores[indices]
    labels = labels[indices]
    # Convert boxes to DeepSort format (x_center, y_center, width, height)
    for box, score, label in zip(boxes, scores, labels):
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        formatted_bboxes.append((x_center, y_center, width, height))
        confidences.append(score.item())
        class_ids.append(label)

    return formatted_bboxes, confidences, class_ids

def tracking_deepsort(predictions, config, video_name):
    if predictions is None:
        print('no trackable objects detected, returning None')
        return None
    
    buffer = []
    buffer_size = 5
    tracker = DeepSortTracker("./library/deep_sort_pytorch/configs/deep_sort.yaml")
    video_path = os.path.join(config['inputs'], video_name)
    frames = get_frames_from_video(video_path)
    all_tracking_results = []
    count = 0
    for frame, prediction in zip(frames, predictions):
        

        count+=1
        deepsort_bboxes, deepsort_confidences, deepsort_class_ids = format_for_deepsort(prediction, threshold=0.75)
        

        if len(deepsort_bboxes) == 0:
            # If there are no objects to track, append a dictionary with the same keys as the original prediction but with empty tensors as values
            all_tracking_results.append({**{key: torch.tensor([]) for key in prediction.keys()}, 'ids': torch.tensor([])})
            continue
        outputs = tracker.update(deepsort_bboxes, deepsort_confidences, deepsort_class_ids, frame)

        if len(outputs) > 0:
            # Initialize tracking_results as an empty list
            tracking_results = []
            bbox_xyxy = outputs[0]  # The first array contains bounding box coordinates
            identities = outputs[1]  # The second array contains identities
            object_ids = outputs[2]  # The third array contains object IDs

            # Since bbox_xyxy, identities, and object_ids are all arrays of the same length,
            # we can loop over them using a single index
            for i in range(len(identities)):
                tracking_results.append([*bbox_xyxy[i], identities[i], object_ids[i]])
            
            # Convert tracking_results to a numpy array
            tracking_results = np.array(tracking_results)

            # Create a new dictionary for the tracked prediction
            tracked_prediction = {}
            for key in prediction:
                if key == 'boxes':
                    tracked_prediction[key] = torch.from_numpy(tracking_results[:, :4])  # The first 4 columns are the bbox coordinates
                elif key == 'labels':
                    tracked_prediction[key] = torch.from_numpy(tracking_results[:, 5].astype(int))  # The 6th column is the class ID
                else:
                    # Calculate IoU between tracked boxes and original boxes
                    iou = box_iou(torch.from_numpy(tracking_results[:, :4]), prediction['boxes'])
                    # Find the index of the original box with the highest IoU for each tracked box
                    indices = torch.argmax(iou, dim=1)
                    # Only include the values corresponding to the tracked objects
                    tracked_prediction[key] = prediction[key][indices]

            # Add the track IDs as a new key
            tracked_prediction['ids'] = torch.from_numpy(tracking_results[:, 4].astype(int))  # The 5th column is the object ID
            buffer.append((frame, tracked_prediction))

            if len(buffer) > buffer_size:
                _, tracked_prediction = buffer.pop(0)
                all_tracking_results.append(tracked_prediction)
        else:
            # If there are no objects to track, append a dictionary with the same keys as the original prediction but with empty tensors as values
            all_tracking_results.append({**{key: torch.tensor([]) for key in prediction.keys()}, 'ids': torch.tensor([])})

    # After all frames have been processed, add the remaining frames in the buffer to all_tracking_results
    for _, tracked_prediction in buffer:
        all_tracking_results.append(tracked_prediction)
    
    return all_tracking_results
  

def tracking(predictions, config, video_name):
    """
    Applies the SORT algorithm to track instances across frames.

    Args:
        predictions (list of dicts): A list of dictionaries containing the predictions made by the model.

    Returns:
        list of dicts: A list of dictionaries containing the tracked instances.
    """
    # Initialize SORT tracker
    tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.5)
    print('Tracking')
    # Count how many frames have detected persons in them
    frames_with_persons = sum(1 for prediction in predictions if len(prediction['boxes']) > 0)
    print(f"Number of frames with detected persons: {frames_with_persons}")

    tracked_predictions = []
    frames_with_no_object = 0
    for prediction in predictions:
        if len(prediction['boxes']) == 0:
            frames_with_no_object += 1
            print('one frame with no person')
            if frames_with_no_object == 150:
                print('No boxes in the 150 frames, stopping tracking')
                return None
            else:
                continue  # Skip this frame and continue with the next one
        
        # Reset the counter if a person is detected

        bboxes_scores = np.column_stack((prediction['boxes'], prediction['scores']))
        tracked_bboxes = tracker.update(bboxes_scores)
        if tracked_bboxes.size == 0:
            print("No boxes to be tracked in this frame, skipping")
            continue  # Skip this frame and continue with the next one
        new_boxes = tracked_bboxes[:, :4]
        ids = tracked_bboxes[:, 4]
        score_identifier = tracked_bboxes[:, 5]
        # Convert the tensor to a numpy array
        scores_np = prediction['scores'].numpy()
        tracked_prediction = {}
        tracked_prediction['scores'] = torch.from_numpy(score_identifier)
        # Create a mapping from old scores to their indices
        score_to_index = {score: i for i, score in enumerate(scores_np)}
        # For each key in prediction that is not 'boxes' or 'scores'
        for key in prediction:
            if key not in ['boxes', 'scores']:  
                tensor_list = [prediction[key][score_to_index[score.item()]] for score in tracked_prediction['scores']]
                tracked_prediction[key] = torch.stack(tensor_list)
                if len(tensor_list) > 1:
                    tracked_prediction[key] = tracked_prediction[key].squeeze()
        tracked_prediction['boxes'] = torch.from_numpy(new_boxes)
        tracked_prediction['ids'] = torch.from_numpy(ids.astype(int))
        tracked_prediction['scores'] = torch.from_numpy(score_identifier)
        tracked_predictions.append(tracked_prediction)
    if len(tracked_predictions)  <20:
        return None
    
    return tracked_predictions


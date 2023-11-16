import csv
import os 
def save_segmentation_as_csv(segmentation, filename):
    """
    Save the semantic segmentation output as a CSV file.

    Args:
        segmentation (list): A list of 2D arrays representing the segmentation maps for each frame.
        filename (str): The name of the CSV file to save.
    """

    # Initialize an empty list to store the rows of the CSV file
    rows = []

    # For each frame...
    for frame_index, frame in enumerate(segmentation):
        # For each pixel in the frame...
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                # Get the class label of the pixel
                class_label = frame[y, x]

                # Create a dictionary with 'frame', 'x', 'y', and 'class' as keys and the corresponding values
                row = {'frame': frame_index, 'x': x, 'y': y, 'class': class_label}

                # Append the dictionary to the list
                rows.append(row)

    # Write the list of dictionaries to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'x', 'y', 'class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

import csv
import numpy as np

def save_keypoints_as_csv(predictions, filename):
    """
    Save the keypoints predictions as a CSV file.

    Args:
        predictions (list): A list of dictionaries containing the keypoints predictions for each frame.
        filename (str): The name of the CSV file to save.
    """

    # Initialize an empty list to store the rows of the CSV file
    rows = []

    # For each prediction...
    for frame_index, prediction in enumerate(predictions):
        # Convert the keypoints tensor to a numpy array
        keypoints = prediction['keypoints'].numpy()

        # For each instance in the prediction...
        for instance_index, instance in enumerate(keypoints):
            # For each keypoint in the instance...
            for keypoint in instance:
                # Create a dictionary with 'frame', 'id', 'keypoint', 'class', 'box', 'score', and 'keypoint_score' as keys and the corresponding values
                row = {
                    'frame': frame_index, 
                    'id': instance_index, 
                    'keypoint': (keypoint[0], keypoint[1], keypoint[2]), 
                    'class': prediction['labels'][instance_index],
                    'box': prediction['boxes'][instance_index],
                    'score': prediction['scores'][instance_index],
                    'keypoint_score': prediction['keypoint_scores'][instance_index]
                }

                # Append the dictionary to the list
                rows.append(row)

    # Write the list of dictionaries to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'id', 'keypoint', 'class', 'box', 'score', 'keypoint_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




def determine_and_execute_export_function(data_dict, output_path):
    """
    Determines and executes the appropriate exporting function for each model in the data_dict.
    Args:
        data_dict (dict): A dictionary containing the data to be exported.
        output_path (str): The base directory to store the exported files.
    """
    for model_name, model_data in data_dict.items():
        for video_name, data in model_data.items():
            # Create a directory for the model if it doesn't exist
            model_dir = os.path.join(output_path, video_name, model_name)
            os.makedirs(model_dir, exist_ok=True)

            if model_name == 'semantic_segmentation':
                # Convert the segmentation scores to class labels
                class_labels = data.argmax(dim=1)
                # Convert the tensor of class labels to a list of 2D numpy arrays
                segmentation_list = [frame.numpy() for frame in class_labels]
                # Save the segmentation to a CSV file
                save_segmentation_as_csv(segmentation_list, os.path.join(model_dir, f"{video_name}_{model_name}.csv"))
            elif model_name == 'keypoints':
                save_keypoints_as_csv(data, os.path.join(model_dir, f"{video_name}_{model_name}.csv"))
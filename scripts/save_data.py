import csv
import os 
import scripts.visualizer as visualizer
import csv
import numpy as np


import h5py

def save_segmentation_as_hdf5(segmentation, filename):
    """
    Save the semantic segmentation output as an HDF5 file.
    This is done on the data with all probabilities per class on a pixel, 
    maybe we need to decide to first argmax for the dominant class so we just have per pixel what class it belongs to.

    Args:
        segmentation (list): A list of tensors representing the segmentation maps for each frame.
        filename (str): The name of the HDF5 file to save.
    """

    with h5py.File(filename, 'w') as f:
        for i, frame in enumerate(segmentation):
            # Convert the tensor to numpy array
            frame_np = frame.numpy()
            # Create a dataset for each frame in the HDF5 file
            f.create_dataset(f'frame_{i}', data=frame_np)

def keypoints_to_csv(data, output_filename):
    # Define the keypoints classes
    coco_keypoints = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    # Find the maximum number of instances in any frame
    max_instances = max(len(frame_data["ids"]) for frame_data in data)

    # Open the CSV file for writing
    with open(output_filename, 'w', newline='') as csv_file:
        # Initialize the CSV writer
        writer = csv.writer(csv_file)

        # Write the header row
        header = ['Frame', 'Num_Instances']
        for i in range(max_instances):  # For each instance...
            for keypoint in coco_keypoints:  # For each keypoint...
                header.append(f'Instance_{i}_{keypoint}')
        writer.writerow(header)


        # For each frame...
        for frame_index, frame_data in enumerate(data):
            # Initialize the row with the frame index and number of instances
            row = [frame_index, len(frame_data["ids"])]
            # For each instance...
            for instance_index, instance in enumerate(frame_data["keypoints"]):
                # For each keypoint...
                for keypoint in instance:
                    # Convert the keypoint data to a tuple and add it to the row
                    row.append(tuple(keypoint.tolist()))
            # Write the row to the CSV file
            writer.writerow(row)
        


def determine_and_execute_export_function(data_dict, config):
    """
    Determines and executes the appropriate exporting function for each model in the data_dict.
    Args:
        data_dict (dict): A dictionary containing the data to be exported.
        output_path (str): The base directory to store the exported files.
    """
    for task_type, task_data in data_dict.items():
        for model_name, model_data in task_data.items():
            for video_name, data in model_data.items():
                # Create a directory for the model if it doesn't exist
                output_path = config['outputs']
                task_dir = os.path.join(output_path, video_name, task_type, model_name)
                os.makedirs(task_dir, exist_ok=True)

                # Get the export settings for the current task and model
                export_settings = config['tasks'][task_type][model_name]['export']
                print(export_settings)
                if export_settings.get('resize', None):
                    resize_value = export_settings['resize']
                if task_type == 'semantic_segmentation':
                    if export_settings.get('csv', False):
                        # Save the segmentation to a CSV file 
                        save_segmentation_as_hdf5(data, os.path.join(task_dir, f"{video_name}_{model_name}.hdf5"))
                    # Check if video export is requested
                    if export_settings.get('video', False):
                        # Save the segmentation as a video
                        visualizer.create_videos_from_frames(data, os.path.join(task_dir, f"{video_name}_{model_name}.mp4"), task_type, video_name, config, resize_value)
                        print(f'Video exported to {task_dir}/{video_name}_{model_name}.mp4')
                if task_type == 'keypoints':
                    print('here')
                    # Check if CSV export is requested
                    if export_settings.get('csv', False):
                        keypoints_to_csv(data, os.path.join(task_dir, f"{video_name}_{model_name}.csv"))
                    if export_settings.get('video', False):
                        # Save the segmentation as a video
                        print('Creating video')
                        visualizer.create_videos_from_frames(data, os.path.join(task_dir, f"{video_name}_{model_name}.mp4"), task_type, video_name, config, resize_value)
                        print(f'Video exported to {task_dir}/{video_name}_{model_name}.mp4')
                if task_type == "instance_segmentation":
                    # Check if video export is requested
                    if export_settings.get('video', False):
                        # Save the segmentation as a video
                        visualizer.create_videos_from_frames(data, os.path.join(task_dir, f"{video_name}_{model_name}.mp4"), task_type, video_name, config, resize_value)
                        print(f'Video exported to {task_dir}/{video_name}_{model_name}.mp4')
                    if export_settings.get('csv', False):
                        # Save the segmentation to a CSV file 
                        save_segmentation_as_hdf5(data, os.path.join(task_dir, f"{video_name}_{model_name}.hdf5"))
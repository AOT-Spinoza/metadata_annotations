import csv
import os 
import scripts.visualizer as visualizer
import csv
import numpy as np
import pandas as pd
import torch

import h5py



def load_data_from_hdf5(filename):
    """
    Load the object detection output from an HDF5 file.

    Args:
        filename (str): The name of the HDF5 file to load.

    Returns:
        data (list): A list of dictionaries representing the detection maps for each frame.
    """
    data = []

    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            frame = {}
            for subkey in f[key].keys():
                frame[subkey] = np.array(f[key][subkey])
            data.append(frame)

    return data

def save_detection_as_hdf5(data, filename):
    """
    Save the object detection output as an HDF5 file.


    Args:
        detections (list): A list of tensors representing the detection maps for each frame.
        filename (str): The name of the HDF5 file to save.
    """

    for frame in data:
        if frame != None:
            for key, value in frame.items():
                try:
                    frame[key] = value.detach().numpy()
                except:
                    frame[key] = value

    with h5py.File(filename, 'w') as f:
        for i, d in enumerate(data):
            grp = f.create_group(str(i))
            try:
                for key, value in d.items():
                    grp.create_dataset(key, data=value, compression="gzip", compression_opts=9)
            except AttributeError:
                # Create an empty dataset for frames with no detections
                grp.create_dataset("empty", data=np.array([]), compression="gzip", compression_opts=9)

def save_instance_segmentation_as_hdf5(data, filename):
    """
    Save the instance segmentation output as an HDF5 file.

    Args:
        segmentation (list): A list of dictionaries representing the segmentation maps for each frame.
        filename (str): The name of the HDF5 file to save.
    """
    for frame in data:
        for key, value in frame.items():
            try:
                frame[key] = value.detach().numpy()
            except:
                frame[key] = value
    # Write to HDF5 file
    with h5py.File(filename, 'w') as f:
        for i, d in enumerate(data):
            grp = f.create_group(str(i))
            for key, value in d.items():
                grp.create_dataset(key, data=value, compression="gzip", compression_opts=9)






def save_instance_segmentation_as_hdf5_memory(data, filename):
    """
    Save the instance segmentation output as an HDF5 file.

    Args:
        data (list): A list of dictionaries representing the segmentation maps for each frame.
        filename (str): The name of HDF5 file save.
    """
    
        
    with h5py.File(filename, 'w') as f:
        # Initialize height and width
        height, width = None, None

        # Find the first non-empty 'masks' to get the height and width
        for frame in data[2:]:
            try:
                height, width = frame['masks'][0].shape[1:]
                break
            except:
                continue
        if height == None or width == None:
            print('No masks found in frames')
            return
        for i, frame in enumerate(data[2:], start=2):  # Start from the third frame
            # Initialize an empty 2D array for the instance map

            instance_map = np.zeros((height, width), dtype=np.uint16)

            # Initialize an empty dictionary for the instance labels
            instance_labels = {}

            for mask, label, id in zip(frame['masks'], frame['labels'], frame['ids']):
                # Convert the probability mask to a boolean mask
                bool_mask = mask[0] > 0.5

                # Assign the instance ID to the pixels in the mask
                instance_map[bool_mask] = id

                # Map the instance ID to the label
                instance_labels[id] = label.item()

            # Save the instance map and the instance labels for this frame
            grp = f.create_group(str(i))
            grp.create_dataset('instance_map', data=instance_map, compression="gzip", compression_opts=9)
            grp.create_dataset('instance_labels', data=np.array(list(instance_labels.items()), dtype=np.uint16), compression="gzip", compression_opts=9)
            grp.attrs['frame_number'] = i  # Store the frame number as an attribute

def save_segmentation_as_hdf5(segmentation, filename):
    """
    Save the semantic segmentation output as an HDF5 file.
    This version applies argmax to each frame before saving, reducing the data size.

    Args:
        segmentation (list): A list of tensors representing the segmentation maps for each frame.
        filename (str): The name of the HDF5 file to save.
    """

    with h5py.File(filename, 'w') as f:
        for i, frame in enumerate(segmentation):
            # Apply argmax to the frame to get the class with the highest probability for each pixel
            frame_argmax = torch.argmax(frame, dim=0).numpy()
            # Save the frame to the HDF5 file
            f.create_dataset(f'frame_{i}', data=frame_argmax, compression="gzip", compression_opts=9)
        print(f'HDF5 exported to {filename}')

def save_motion_energy_as_hdf5(moten_features, filename):
    """
    Save the semantic segmentation output as an HDF5 file.
    This version applies argmax to each frame before saving, reducing the data size.

    Args:
        segmentation (list): A list of tensors representing the segmentation maps for each frame.
        filename (str): The name of the HDF5 file to save.
    """

    with h5py.File(filename, 'w') as f:
        for i, frame in enumerate(moten_features):
            # Save the frame to the HDF5 file
            f.create_dataset(f'frame_{i}', data=frame, compression="gzip", compression_opts=9)
        print(f'HDF5 exported to {filename}')

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
        


def determine_and_execute_export_function(data_dict,classes_dict, config):
    """
    Determines and executes the appropriate exporting function for each model in the data_dict.
    Args:
        data_dict (dict): A dictionary containing the data to be exported.
        output_path (str): The base directory to store the exported files.
    """
    for task_type, task_data in data_dict.items():
        for model_name, model_data in task_data.items():
            if classes_dict[task_type][model_name] != None:
                classes = classes_dict[task_type][model_name]
            if config['tasks'][task_type][model_name]["load_model"]['framework'] in ("torchhub", "huggingface"):
                export_settings = config['tasks'][task_type][model_name]['export']
                resize_value =  export_settings.get('resize', None)
                print(export_settings)
                if export_settings.get('csv', None):
                    # Save the segmentation to a CSV file 
                    for video_name, data in model_data.items():
                        output_path = config['outputs']
                        task_dir = os.path.join(output_path, video_name, task_type, model_name)
                        os.makedirs(task_dir, exist_ok=True)
                        if model_name == 'X3D':
                            with open(os.path.join(task_dir, f"{video_name}_{model_name}.csv"), 'w', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=['pred_class', 'pred_value'])
                                writer.writeheader()
                                writer.writerow(data)
                            print(f'CSV exported to {task_dir}/{video_name}_{model_name}.csv')
                        if model_name == "GIT":
                            with open(os.path.join(task_dir, f"{video_name}_{model_name}.csv"), 'w', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(['Caption'])
                                writer.writerow([data['caption']])
                            print(f'CSV exported to {task_dir}/{video_name}_{model_name}.csv')
                              
                 
                if export_settings.get('hdf5', False):
                    for video_name, data in model_data.items():
                        output_path = config['outputs']
                        task_dir = os.path.join(output_path, video_name, task_type, model_name)
                        os.makedirs(task_dir, exist_ok=True)
                        if model_name == "MiDaS":
                            # Convert each tensor to a numpy array
                            data_np = [frame.numpy() for frame in data]
                            # Write to HDF5 file
                            with h5py.File(os.path.join(task_dir, f"{video_name}_{model_name}.hdf5"), 'w') as f:
                                # Create a 3D dataset for all frames in the HDF5 file
                                f.create_dataset('frames', data=data_np, compression="gzip", compression_opts=9)
                                print(f'HDF5 exported to {task_dir}/{video_name}_{model_name}.hdf5')
                            
                if export_settings.get('video', False):
                    for video_name, data in model_data.items():
                        output_path = config['outputs']
                        task_dir = os.path.join(output_path, video_name, task_type, model_name)
                        video_name = os.path.splitext(video_name)[0]
                        os.makedirs(task_dir, exist_ok=True)
                        if model_name == "MiDaS":
                            # Save the depth map as a video
                            visualizer.create_videos_from_frames(data, os.path.join(task_dir, f"{video_name}_{model_name}.mp4"), task_type, video_name, config, resize_value)
                            print(f'Video exported to {task_dir}/{video_name}_{model_name}.mp4')

            if config['tasks'][task_type][model_name]["load_model"]['framework'] in ("torch", "pytorchvideo"):
                
                for video_name, data in model_data.items():
                    
                    # Create a directory for the model if it doesn't exist
                    output_path = config['outputs']
                    task_dir = os.path.join(output_path, video_name, task_type, model_name)
                    video_name = os.path.splitext(video_name)[0]
                    os.makedirs(task_dir, exist_ok=True)
                    # Get the export settings for the current task and model
                    export_settings = config['tasks'][task_type][model_name]['export']
                    
                    resize_value =  export_settings.get('resize', None)
                    if task_type == 'semantic_segmentation':
                        if export_settings.get('csv', False):
                            # Save the segmentation to a CSV file 
                            save_segmentation_as_hdf5(data, os.path.join(task_dir, f"{video_name}_{model_name}.hdf5"))
                        # Check if video export is requested
                        if export_settings.get('video', False):
                            print('Creating video')
                            # Save the segmentation as a video
                            visualizer.create_videos_from_frames(data, os.path.join(task_dir, f"{video_name}_{model_name}.mp4"), task_type, video_name, config, resize_value, classes)
                            print(f'Video exported to {task_dir}/{video_name}_{model_name}.mp4')
                        if export_settings.get('hdf5', False):
                            # Save the segmentation to a CSV file 
                            save_segmentation_as_hdf5(data, os.path.join(task_dir, f"{video_name}_{model_name}.hdf5"))
                    if task_type == 'keypoints':
                        if data != None:
                            # Check if CSV export is requested
                            if export_settings.get('csv', False):
                                    keypoints_to_csv(data, os.path.join(task_dir, f"{video_name}_{model_name}.csv"))
                            if export_settings.get('hdf5', False):
                                    save_detection_as_hdf5(data, os.path.join(task_dir, f"{video_name}_{model_name}.hdf5"))                         
                            if export_settings.get('video', False):
                                # Save the segmentation as a video
                                print('Creating video')
                                visualizer.create_videos_from_frames(data, os.path.join(task_dir, f"{video_name}_{model_name}.mp4"), task_type, video_name, config, resize_value, classes)
                                print(f'Video exported to {task_dir}/{video_name}_{model_name}.mp4')
                        else:
                            with open(os.path.join(task_dir, f"{video_name}_{model_name}.txt"), 'w') as f:
                                f.write("No keypoints detected in video")
                    if task_type == "instance_segmentation":
                        # Check if video export is requested
                        if export_settings.get('video', False):
                            # Save the segmentation as a video
                            print('Creating video')
                            x = visualizer.create_videos_from_frames(data, os.path.join(task_dir, f"{video_name}_{model_name}.mp4"), task_type, video_name, config, resize_value, classes)
                            if x ==1:
                                print(f'Video exported to {task_dir}/{video_name}_{model_name}.mp4')
                            else:
                                print(f'No video exported')
                        if export_settings.get('hdf5', False):
                            # Save the segmentation to a CSV file 
                            save_instance_segmentation_as_hdf5_memory(data, os.path.join(task_dir, f"{video_name}_{model_name}.hdf5")) 
                            print(f'HDF5 exported to {task_dir}/{video_name}_{model_name}.hdf5')
                    if task_type == "object_detection":
                        if data != None:
                            # Check if video export is requested
                            if export_settings.get('video', False):
                                # Save the segmentation as a video
                                print('Creating video')
                                visualizer.create_videos_from_frames(data, os.path.join(task_dir, f"{video_name}_{model_name}.mp4"), task_type, video_name, config, resize_value, classes)
                                print(f'Video exported to {task_dir}/{video_name}_{model_name}.mp4')
                            if export_settings.get('hdf5', False):
                                save_detection_as_hdf5(data, os.path.join(task_dir, f"{video_name}_{model_name}.hdf5")) 
                        else:
                            with open(os.path.join(task_dir, f"{video_name}_{model_name}.txt"), 'w') as f:
                                f.write("No objects detected in video")
                    if task_type == "action_detection":
                        if data != None:
                            # Check if video export is requested
                            if export_settings.get('video', False):
                                # Save the action detection as a video
                                print('Creating video')
                                visualizer.create_videos_from_frames(data, os.path.join(task_dir, f"{video_name}_{model_name}.mp4"), task_type, video_name, config, resize_value, classes)
                                print(f'Video exported to {task_dir}/{video_name}_{model_name}.mp4')
                            if export_settings.get('hdf5', False):
                                save_detection_as_hdf5(data, os.path.join(task_dir, f"{video_name}_{model_name}.hdf5"))
                        else:
                            with open(os.path.join(task_dir, f"{video_name}_{model_name}.txt"), 'w') as f:
                                f.write("No action detected in video")
            if config['tasks'][task_type][model_name]["load_model"]['framework'] == 'pymoten':
                export_settings = config['tasks'][task_type][model_name]['export']
                for video_name, data in model_data.items():
                    output_path = config['outputs']
                    task_dir = os.path.join(output_path, video_name, task_type, model_name)
                    os.makedirs(task_dir, exist_ok=True)
                    video_name = os.path.splitext(video_name)[0]
                    if export_settings.get('hdf5', False):
                        save_motion_energy_as_hdf5(data, os.path.join(task_dir, f"{video_name}_{model_name}.hdf5"))

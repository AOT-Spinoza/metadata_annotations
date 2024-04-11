import h5py
import numpy as np
from scipy.sparse import csr_matrix

def load_depth_maps_from_hdf5(filename):
    """
    Load the depth maps from an HDF5 file.

    Args:
        filename (str): The name of the HDF5 file to load.

    Returns:
        depth_map (numpy.ndarray): A 3D numpy array representing the depth map.
    """
    with h5py.File(filename, 'r') as f:
        depth_map = np.array(f['frames'])
    return depth_map

def load_keypoints_from_hdf5(filename):
    """
    Load the keypoints from an HDF5 file.

    Args:
        filename (str): The name of the HDF5 file to load.

    Returns:
        keypoints (list of dict): A list of dictionaries representing the keypoints.
    """
    keypoints = []
    with h5py.File(filename, 'r') as f:
        for group_name in f:
            group = f[group_name]
            frame = {key: np.array(dataset) for key, dataset in group.items()}
            keypoints.append(frame)
    return keypoints

def load_instance_segmentation_from_hdf5_memory(filename):
    """
    Load the instance segmentation output from an HDF5 file.

    Args:
        filename (str): The name of the HDF5 file to load.

    Returns:
        data (list): A list of dictionaries representing the segmentation maps for each frame.
    """
    data = []

    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            # Load the instance map and the instance labels for this frame
            grp = f[key]
            instance_map = np.array(grp['instance_map'])
            instance_labels = {id: label for id, label in grp['instance_labels']}

            # Create a dictionary for this frame
            frame = {
                'instance_map': instance_map,
                'instance_labels': instance_labels,
                'frame_number': grp.attrs['frame_number']
            }

            # Add the frame to the data list
            data.append(frame)

    return data



def load_segmentation_from_hdf5(filename):
    """
    Load the semantic segmentation output from an HDF5 file and convert it to sparse matrices.

    Args:
        filename (str): The name of the HDF5 file to load.

    Returns:
        list: A list of sparse matrices representing the segmentation maps for each frame.
    """

    with h5py.File(filename, 'r') as f:
        # Load each frame from the HDF5 file and convert it to a sparse matrix
        segmentation = [csr_matrix(f[f'frame_{i}']) for i in range(len(f.keys()))]

    return segmentation

def load_objects_from_hdf5(filename):
    """
    Load the keypoints from an HDF5 file.

    Args:
        filename (str): The name of the HDF5 file to load.

    Returns:
        keypoints (list of dict): A list of dictionaries representing the keypoints.
    """
    data = []
    with h5py.File(filename, 'r') as f:
        for group_name in f:
            group = f[group_name]
            frame = {key: np.array(dataset) for key, dataset in group.items()}
            data.append(frame)
    return data

import h5py
import numpy as np

def load_objects_from_hdf5_2(filename):
    """
    Load the object detection output from an HDF5 file.

    Args:
        filename (str): The name of the HDF5 file to load.

    Returns:
        data (list): A list of dictionaries representing the detection maps for each frame.
    """

    data = []

    with h5py.File(filename, 'r') as f:
        for i in range(len(f.keys())):
            grp = f[str(i)]
            frame_data = {}
            for key in grp.keys():
                if key == "empty":
                    frame_data = None
                else:
                    frame_data[key] = np.array(grp[key])
            data.append(frame_data)

    return data
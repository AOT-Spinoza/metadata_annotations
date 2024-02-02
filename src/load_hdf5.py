import h5py
import numpy as np

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
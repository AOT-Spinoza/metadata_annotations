
def find_video_path(video_name, config):
    """
    Find the path of a video file based on the video name and configuration.

    Args:
        video_name (str): The name of the video file.
        config (dict): The configuration containing the input directory.

    Returns:
        str: The path of the video file.

    """
    # Construct the video path using the input directory from the config and video name
    video_path = f"{config['inputs']}/{video_name}"
    
    return video_path

import itertools
import torch
import torchvision.io

def custom_read_video(video_object, start=0, end=None, read_video=True, read_audio=False):
    """
    Reads a video file and returns the video frames, audio frames, video and audio pts, and metadata.

    Args:
        video_object (object): A video object.
        start (float): The start time in seconds. Default is 0.
        end (float): The end time in seconds. Default is None.
        read_video (bool): Whether to read video frames. Default is True.
        read_audio (bool): Whether to read audio frames. Default is False.

    Returns:
        tuple: A tuple containing:
            - video_frames (torch.Tensor): A tensor containing the video frames in the format (T, C, H, W).
            - audio_frames (torch.Tensor): A tensor containing the audio frames.
            - video_pts (list): A list of video pts.
            - audio_pts (list): A list of audio pts.
            - metadata (dict): A dictionary containing the metadata.

    The function reads a video file and returns the video frames, audio frames, video and audio pts, and metadata.
    The video frames and audio frames are returned as tensors, while the video and audio pts are returned as lists.
    The metadata is returned as a dictionary.

    Example usage:
    ```
    video_object = torchvision.io.VideoReader("video.mp4")
    video_frames, audio_frames, video_pts, audio_pts, metadata = custom_read_video(video_object)
    ```

    The video_frames tensor has the following dimensions:
    - T: number of frames in the video
    - C: number of channels in the video (3 for RGB)
    - H: height of each frame in pixels
    - W: width of each frame in pixels
    """
    if end is None:
        end = float("inf")
    if end < start:
        raise ValueError(
            "end time should be larger than start time, got "
            f"start time={start} and end time={end}"
        )

    video_frames = torch.empty(0)
    video_pts = []
    if read_video:
        video_object.set_current_stream("video")
        frames = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end, video_object.seek(start)):
            # Add the color channel dimension to the frame
            frame_with_channel = frame['data']
            frames.append(frame_with_channel)
            video_pts.append(frame['pts'])
        if len(frames) > 0:
            video_frames = torch.stack(frames, 0)
    audio_frames = torch.empty(0)
    audio_pts = []
    if read_audio:
        video_object.set_current_stream("audio")
        frames = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end, video_object.seek(start)):
            frames.append(frame['data'])
            video_pts.append(frame['pts'])
        if len(frames) > 0:
            audio_frames = torch.cat(frames, 0)

    return video_frames, audio_frames, (video_pts, audio_pts), video_object.get_metadata()

import numpy as np

def get_frames_by_indices(video_object, indices):
    """
    Reads a video file and returns the specified frames.

    Args:
        video_object (object): A video object.
        indices (list): A list of frame indices.

    Returns:
        np.ndarray: A numpy array containing the specified frames in the format (num_frames, height, width, 3).
    """
    video_object.set_current_stream("video")
    frames = []
    indices = sorted(indices)
    idx = 0
    for i, frame in enumerate(video_object):
        if i == indices[idx]:
            # Convert the frame to a numpy array and add it to the list
            frame_np = frame['data'].permute(1, 2, 0).numpy()
            frames.append(frame_np)
            idx += 1
            if idx == len(indices):
                break
    if len(frames) > 0:
        video_frames = np.stack(frames)
    else:
        video_frames = np.empty((0, frame['data'].shape[1], frame['data'].shape[2], 3))
    return video_frames
import numpy as np
import torchvision
from tqdm import tqdm
import torch
from src.video_reader import custom_read_video
import gc
import os

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    
    Args:
        clip_len (int): Total number of frames to sample.
        frame_sample_rate (int): Sample every n-th frame.
        seg_len (int): Maximum allowed index of sample's last frame.
        
    Returns:
        indices (List[int]): List of sampled frame indices
    '''
    # Calculate the length of the clip after applying the frame sample rate
    converted_len = int(clip_len * frame_sample_rate)
    
    # Randomly select an end index for the clip
    end_idx = np.random.randint(converted_len, seg_len)
    
    # Calculate the start index of the clip
    start_idx = end_idx - converted_len
    
    # Generate a list of frame indices
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    
    # Clip the indices to ensure they are within the valid range
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    
    return indices

def infer_videos_huggingface(video_files, model, transformations, model_name):
    '''
    Process a list of videos using a Hugging Face model.
    
    Args:
        video_files (List[str]): List of paths to the video files.
        model (transformers.PreTrainedModel): The Hugging Face model to use.
        transformations (transformers.PreTrainedTokenizer): The tokenizer to use for the model.
        model_name (str): The name of the model.
        
    Returns:
        outputs_all (Dict[str, str]): A dictionary mapping video file names to model outputs.
    '''
    # Initialize the output dictionary
    outputs_all = {}
    # Set the random seed for reproducibility
    seed = np.random.seed(40)
    
    # Move the model to the GPU if available
    model = model.to('cuda')
    
    # Process each video file
    for video in tqdm(video_files, desc=f"Processing videos for {model_name}"):
        # Create a VideoReader for the video file
        video_reader = torchvision.io.VideoReader(video, "video")
        
        # Read the video frames using a custom function
        video_frames, _, pts, meta = custom_read_video(video_reader)

        # Delete the VideoReader to free up memory
        del video_reader
        
        # Run the garbage collector to free up memory
        gc.collect()
        
        # Get the number of frames to sample from the model's configuration
        num_frames = model.config.num_image_with_embedding
        
        # Sample frame indices
        indices = sample_frame_indices(clip_len=num_frames, frame_sample_rate=20, seg_len=len(video_frames))   
        numpy_array = np.array(indices)
        # Convert the list of indexes to a PyTorch tensor
        indexes = torch.tensor(indices)
        
        # Select the frames at the sampled indices
        selected_frames = video_frames[indexes] # (T, C, H, W)
        
        # Convert the selected frames to a numpy array
        selected_frames = selected_frames.numpy()
        
        # Change the dimensions of the frames to (T,H, W, C)
        selected_frames = selected_frames.transpose((0, 2, 3, 1))
        
        # Apply the transformations to the frames and convert them to tensors
        pixel_values = transformations(images=list(selected_frames), return_tensors="pt").pixel_values
        
        # Move the pixel values to the GPU if available
        pixel_values = pixel_values.to('cuda')
        
        # Generate predictions using the model
        generated_ids = model.generate(pixel_values=pixel_values, max_length=100)
        
        # Decode the generated IDs to get the output text
        output = transformations.batch_decode(generated_ids, skip_special_tokens=True)

        # Add the output to the dictionary
        outputs_all[os.path.basename(video)] = output
        
    return outputs_all
import os
import torch
from src.video_reader import custom_read_video
import torchvision.io
import collections
import gc
from tqdm import tqdm

def infer_videos(video_files, model, transformation, config, task_type, model_name):
    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval()
    print(f"Using device: {device}")
    # Move the model to the device
    model = model.to(device)
    # Initialize the output dictionary
    outputs_all = {model_name: {}}
    for video in tqdm(video_files, desc=f"Processing videos for {model_name}"):
        video_reader = torchvision.io.VideoReader(video, "video")
        video_frames, _, pts, meta = custom_read_video(video_reader)
        del video_reader
        gc.collect()
        # Initialize the list of predictions for this video
        outputs_all[model_name][os.path.basename(video)] = []
        for frame_count, frame in enumerate(tqdm(video_frames, desc=f"Processing frames for {os.path.basename(video)}", leave=False), start=1):
            # Apply preprocessing steps if they are specified in the config
            if config.tasks[task_type][model_name].preprocessing.unsqueeze:
                frame = frame.unsqueeze(0)
            if config.tasks[task_type][model_name].preprocessing.to_tensor:
                frame = torch.from_numpy(frame)
            
            # Apply the transformation to the video frame.

            transformed_frame = transformation(frame)

            # Move the transformed frame to the device
            transformed_frame = transformed_frame.to(device)
            # Perform inference
            if config.tasks[task_type][model_name].preprocessing.to_list:
                transformed_frame = [transformed_frame]
            with torch.no_grad():
                output = model(transformed_frame)
            if isinstance(output, collections.OrderedDict):
                output = output['out']
            # Detach the output from the computation graph and move it to CPU
            if isinstance(output, torch.Tensor):
                output = output.detach().cpu()
            else:
                output = [{k: v.detach().cpu() for k, v in dict_.items()} for dict_ in output]
            # Store the output in the output list.
            outputs_all[model_name][os.path.basename(video)].append(output)
    del model
    torch.cuda.empty_cache()
    return outputs_all
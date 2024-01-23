from pytorchvideo.data.encoded_video import EncodedVideo
from tqdm import tqdm
import torch
from scripts.visualizer import write_video
import os
from src.video_reader import custom_read_video
import torchvision.io
import collections
import gc
import matplotlib.pyplot as plt
import cv2

def infer_videos_torchhub(video_files, model, transformation, clip_duration, classes, model_name):
    """
    Perform inference on a list of video files using a TorchHub model.

    Args:
        video_files (list): List of video file paths.
        model : TorchHub model for inference.
        transformation: Transformation to apply to video frames.
        clip_duration (float): Duration of video clips to process.
        classes (list): List of class labels.
        model_name (str): Name of the TorchHub model.

    Returns:
        dict: Dictionary containing the inference results for each video file.
            The keys are the video file names, and the values are dictionaries
            with keys "pred_class" and "pred_value" representing the predicted
            class names and values, respectively, for the X3D model.
            For the MiDaS model, the values are lists of depth maps for each frame in the video.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval()
    print(f"Using device: {device}")
    # Move the model to the device
    model = model.to(device)
    # Initialize the output dictionary
    outputs_all = {}
    try:
        start_sec = 0
        end_sec = start_sec + clip_duration
    except:
        print('clip_duration not found')
    if model_name == "X3D":
        for video_path in tqdm(video_files, desc=f"Processing videos for {model_name}"):
            # Initialize an EncodedVideo helper class and load the video
            video = EncodedVideo.from_path(video_path)

            # Load the desired clip
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
            
            inputs = transformation(video_data)
            inputs = inputs["video"]
            # Move the inputs to the desired device

            inputs = inputs.to(device)

            preds = model(inputs[None, ...])

            # Get the predicted classes


            topk_preds = preds.topk(k=5)
            pred_classes = topk_preds.indices[0]
            pred_values = topk_preds.values[0]

            # Map the predicted classes to the label names
            pred_class_names = [classes[int(i)] for i in pred_classes]
            outputs_all[os.path.basename(video_path)] = {
                "pred_class": pred_class_names,
                "pred_value": pred_values.tolist()
            }

    elif model_name == "MiDaS":
        batch_size = 5  # Set your desired batch size
        for video in tqdm(video_files, desc=f"Processing videos for {model_name}"):
            video_reader = torchvision.io.VideoReader(video, "video")
            video_frames, _, pts, meta = custom_read_video(video_reader)
            print(video_frames[0].shape)
            del video_reader
            gc.collect()
            # Load the desired clip
            outputs_all[os.path.basename(video)] = []
            for i in range(0, len(video_frames), batch_size):
                batch_frames = video_frames[i:i+batch_size]
                original_size = batch_frames[0].shape[-2:]
                # Convert the tensor to a numpy array and move it to CPU
                batch_frames = [frame.permute(1, 2, 0).cpu().numpy() for frame in batch_frames]
                input_batch = torch.stack([transformation(frame) for frame in batch_frames]).to(device)
                # Flatten the batch dimension
                input_batch = input_batch.view(-1, *input_batch.shape[2:])
                with torch.no_grad():
                    prediction = model(input_batch).cpu()
                    print(prediction.shape)
                    #
                    # prediction = prediction.view(prediction.shape[0], -1, *prediction.shape[2:])
                    prediction = prediction.view(prediction.shape[0], 1, prediction.shape[1], -1)
                    print(prediction.shape)
                    outputs_all[os.path.basename(video)].extend(prediction)


    # elif model_name == "MiDaS":
    #     for video in tqdm(video_files, desc=f"Processing videos for {model_name}"):
    #         video_reader = torchvision.io.VideoReader(video, "video")
    #         video_frames, _, pts, meta = custom_read_video(video_reader)
    #         del video_reader
    #         gc.collect()
    #         # Load the desired clip
    #         outputs_all[os.path.basename(video)] = []
    #         for frame_count, frame in enumerate(tqdm(video_frames, desc=f"Processing frames for {os.path.basename(video)}", leave=False), start=1):
    #             # Convert the tensor to a numpy array and move it to CPU
    #             frame = frame.permute(1, 2, 0).cpu().numpy()
    #             input_batch = transformation(frame).to(device)
    #             with torch.no_grad():
    #                 prediction = model(input_batch).cpu()

    #                 #### ONLY FOR FFT
    #                 #prediction =prediction.unsqueeze(1)

    #                 ## needs to go to postprocessing
    #                 prediction = torch.nn.functional.interpolate(
    #                      prediction.unsqueeze(1),
    #                      size=video_frames.shape[-2:],
    #                      mode="bicubic",
    #                      align_corners=False,
    #                  )
                    
    #                 outputs_all[os.path.basename(video)].append(prediction)
            

    else:
        raise NotImplementedError("This model variant is not supported or implemented yet")
    return outputs_all
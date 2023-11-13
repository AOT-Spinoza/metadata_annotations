from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
import torchvision.io
import torch
import numpy as np
import os

import pickle
import json
import sys
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from src.video_reader import custom_read_video
weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()

model = keypointrcnn_resnet50_fpn(weights=weights, progress=True)
model = model.eval()
stream = "video"
video_reader = torchvision.io.VideoReader('/tank/tgn252/vid_folder/R_S_b6-4.mp4', stream)

video_frames, _, pts, meta = custom_read_video(video_reader)
# Read the first frame of the video
frame = video_frames[0]
frame_float = convert_image_dtype(frame, dtype=torch.float)

# Convert the frame to a floating-point tensor
frame_float = transforms(frame_float)

# Pass the frame to the model
outputs = model([frame_float])

# Print the outputs
print(outputs)

print(model([transforms(frame)]))

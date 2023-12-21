# This file includes code adapted from PyTorchVideo https://github.com/facebookresearch/pytorchvideo
# Copyright 2021  Facebook Research
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from torchvision.models import get_weight
import torch

from torchvision.transforms import Compose, Lambda, CenterCrop, Normalize 
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    
)
from torchvision.transforms import v2 as T
from torchvision.transforms._functional_video import normalize

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)


class NormalizeVideo(T.Normalize):
    def forward(self, tensor):
        # tensor shape: (C, T, H, W)
        C, T, H, W = tensor.shape
        tensor = tensor.view(C*T, H, W)
        tensor = super().forward(tensor)
        return tensor.view(C, T, H, W)
    



## TODO: MAKE A NEW TRANSFORM BECAUSE OF DEPRECATING FUNCTIONS IN OLD ONE
# def ava_slowfast_transform_v2(
#     clip,
#     boxes,
#     num_frames = 32, 
#     crop_size = 256,
#     data_mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1),
#     data_std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1),
#     slow_fast_alpha = 4, 
# ):
#     boxes = np.array(boxes)
#     ori_boxes = boxes.copy()

#     height, width = clip.shape[2], clip.shape[3]
#     boxes = clip_boxes_to_image(boxes, height, width)
#     print(f"Before uniform, clip shape: {clip.shape}")
#     clip = UniformTemporalSubsample(num_frames)(clip)
#     clip = clip.float() / 255.0
#     print(f"Before short, clip shape: {clip.shape}")
#     clip, boxes = short_side_scale_with_boxes(clip, boxes, size=crop_size)
#     print(f"Before normalization, clip shape: {clip.shape}")
#     clip = NormalizeVideo(mean=data_mean, std=data_std)(clip)
#     clip = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])

#     # Incase of slowfast, generate both pathways
#     if slow_fast_alpha is not None:
#         fast_pathway = clip
#         # Perform temporal sampling from the fast pathway.
#         slow_pathway = torch.index_select(
#             clip,
#             1,
#             torch.linspace(
#                 0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
#             ).long(),
#         )
#         clip = [slow_pathway, fast_pathway]

#     return clip, torch.from_numpy(boxes), ori_boxes

def ava_slowfast_transform_deprecated(
    clip,
    boxes,
    num_frames = 32, 
    crop_size = 256,
    data_mean = [0.45, 0.45, 0.45],
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, 
):

    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )

    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )

    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )

    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]

    return clip, torch.from_numpy(boxes), ori_boxes
    

def torchhub_transform(torchhub_model_variant, config):

    if torchhub_model_variant == "slowfast_r50_detection":
        transform = ava_slowfast_transform_deprecated
        clip_duration = 0.9
    

    elif torchhub_model_variant == "x3d_s":
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        frames_per_second = 30
        model_transform_params  = {
            "x3d_xs": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 4,
                "sampling_rate": 12,
            },
            "x3d_s": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 13,
                "sampling_rate": 6,
            },
            "x3d_m": {
                "side_size": 256,
                "crop_size": 256,
                "num_frames": 16,
                "sampling_rate": 5,
            }
        }

        # Get transform parameters based on model
        transform_params = model_transform_params[torchhub_model_variant]

        # Note that this transform is specific to the slow_R50 model.
        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(transform_params["num_frames"]),
                    Lambda(lambda x: x/255.0),
                    Normalize(mean, std),
                    ShortSideScale(size=transform_params["side_size"]),
                    CenterCrop(
                    transform_params["crop_size"]
                    )
                ]
            ),
        )

        # The duration of the input clip is also specific to the model.
        clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/frames_per_second
    else:
        raise NotImplementedError("This model variant is not supported or implemented yet")

    return transform, clip_duration

def torch_transform(weights,config):
    weights = get_weight(weights)
    transformation = weights.transforms()
    return transformation, None

def huggingface_transform(weights, config):
    return None, None   
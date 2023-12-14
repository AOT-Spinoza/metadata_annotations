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



from torchvision.transforms import Compose, Lambda, CenterCrop, Normalize
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

def torchhub_transform(torchhub_model_variant, config):
    if torchhub_model_variant == "x3d_s":
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
                        crop_size=(transform_params["crop_size"], transform_params["crop_size"])
                    )
                ]
            ),
        )

        # The duration of the input clip is also specific to the model.
        clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/frames_per_second
    return transform, clip_duration


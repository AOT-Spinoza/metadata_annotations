from omegaconf import OmegaConf
from hydra_zen import to_yaml, builds, instantiate
from src.load_models_zen import load_model
from src.predictions_zen import inference
import os
# Load the configuration file
config = OmegaConf.load('config_zen.yaml')

# Load the models and transformations
# Load the models and transformations
models = {}
transformations = {}
for task_type, task in config.tasks.items():
    models[task_type] = {}
    transformations[task_type] = {}
    for model in task.models:
        # Create a configuration object for the load_model function
        load_model_config = builds(load_model, model_loading_function=model.load_model.model_loading_function, weights=model.load_model.weights, weight_version=model.load_model.weight_version)
        # Call the load_model function with the created configuration
        model_instance, model_transformations = instantiate(load_model_config)
        models[task_type][model.model_name] = model_instance
        transformations[task_type][model.model_name] = model_transformations

input_path = config.inputs.path

# Create a configuration object for the inference function
inference_results = inference(
    models=models,
    transformations=transformations,
    input_dir=input_path,
    config=config
)

# Print the outpu

# Print the output for the 10th frame of the 2nd video
print(f"Output for the 10th frame of the 2nd video: {inference_results['keypoints']['KeypointRCNN_ResNet50']['R_S_video-12.mp4'][10]}")

# Print the classes detected in the first frame of the 1st video
# Get the class with the highest score for each pixel
class_map = inference_results['semantic_segmentation']['FCN_ResNet101']['R_S_bing-www_dailymotion_com_video_x64c56w_164.mp4'][1].argmax(dim=1)

# Get a list of unique classes detected in the image
unique_classes = class_map.unique()

print(f"Unique classes detected in the first frame of the 1st video: {unique_classes}")
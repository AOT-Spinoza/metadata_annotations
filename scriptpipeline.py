from src.load_models import load_model_non_hydra
from src.predictions import inference
from src.postprocessing import postprocess_predictions
from scripts.save_data import determine_and_execute_export_function
from scr.visualizer import visualize
import hydra_zen
from hydra_zen import builds, instantiate

config = hydra_zen.load_from_yaml("/tank/tgn252/metadata_annotations/config_zen.yaml")

def my_pipeline(config):
    # Load the models and transformations.
    models, transformations, weights = load_model_non_hydra(config)
    inputs = config["inputs"]
    outputs = inference(config, models, transformations, inputs)
    postprocessed = postprocess_predictions(outputs, config)
    
    # determine_and_execute_export_function(postprocessed, config["outputs"].get('path'))
    # print('Done!')
    return outputs




inference_results = my_pipeline(config)



# Print the output for the 10th frame of the 2nd video
# for task_type, task_results in inference_results.items():
#     for model_name, model_results in task_results.items():
#         for video_name, video_results in model_results.items():
#             if task_type == 'semantic_segmentation':
#                 # Get the class with the highest score for each pixel
#                 print(type(video_results))
#                 print(video_results.keys())
#                 class_map = video_results['R_S_bing-www_dailymotion_com_video_x64c56w_164.mp4'][0].argmax(dim=1)
#                 # Get a list of unique classes detected in the image
#                 unique_classes = class_map.unique()
#                 unique_classes_list = unique_classes.tolist()
#                 print(f"Unique classes detected in the first frame of {video_name} by {model_name}: {unique_classes_list}")
#             elif task_type == 'keypoints':
#                 print(f"Output for the first frame of {video_name} by {model_name}: {video_results['R_S_bing-www_dailymotion_com_video_x64c56w_164.mp4']}")
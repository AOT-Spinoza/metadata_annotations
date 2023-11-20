from src.load_models import load_model_non_hydra
from src.predictions import inference
from src.postprocessing import postprocess_predictions
from scripts.save_data import determine_and_execute_export_function
from scripts.visualizer import visualize_segmentation
from scripts.visualizer import create_videos_from_frames
import hydra_zen
import numpy as np

config = hydra_zen.load_from_yaml("/tank/tgn252/metadata_annotations/config.yaml")

def my_pipeline(config):
    # Load the models and transformations.
    models, transformations = load_model_non_hydra(config)
    inputs = config["inputs"]
    outputs = inference(config, models, transformations, inputs)
    postprocessed = postprocess_predictions(outputs, config)
    determine_and_execute_export_function(postprocessed, config)



my_pipeline(config)

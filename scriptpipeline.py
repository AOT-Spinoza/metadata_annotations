from src.load_models import load_model_non_hydra
from src.predictions import inference
from src.postprocessing import postprocess_predictions
from scripts.save_data import determine_and_execute_export_function
import hydra_zen
import numpy as np

config = hydra_zen.load_from_yaml("/tank/tgn252/metadata_annotations/config.yaml")

def my_pipeline(config):
    # Load the models and transformations.
    models, transformations, classes, clip_durations = load_model_non_hydra(config)
    inputs = config["inputs"]
    outputs = inference(config, models, transformations, clip_durations, classes, inputs)
    postprocessed = postprocess_predictions(outputs, config)
    print('out of postprocessing')
    determine_and_execute_export_function(postprocessed, config)
    print('out of determine_and_execute_export_function')


my_pipeline(config)


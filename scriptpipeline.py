from src.load_models_and_config import load_all_models_and_metadata
from src.predictions import inference
from src.postprocessing import postprocess_predictions
import src.transformations as transformations
from src.save_data import determine_and_execute_export_function
from scripts.statistical_analysis import execute_all_statistical_analysis
import hydra_zen
import numpy as np

config = hydra_zen.load_from_yaml("/tank/tgn252/metadata_annotations/config.yaml")

def my_pipeline(config):
    """
    Executes a pipeline of operations on the given configuration.

    Args:
        config (dict): The configuration for the pipeline.

    Returns:
        None
    """
    # Load the models and transformations.
    models, transformations, classes, clip_durations = load_all_models_and_metadata(config)
    inputs = config["inputs"]
    outputs = inference(config, models, transformations, clip_durations, classes, inputs)
    postprocessed = postprocess_predictions(outputs, config)
    determine_and_execute_export_function(postprocessed,classes, config)
    print('out of determine_and_execute_export_function')
    # execute_all_statistical_analysis(postprocessed, config)
    


my_pipeline(config)


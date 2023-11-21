from src.load_models import load_model_non_hydra
from src.predictions import inference
from src.postprocessing import postprocess_predictions
from scripts.save_data import determine_and_execute_export_function
import hydra_zen
import numpy as np

config = hydra_zen.load_from_yaml("/tank/tgn252/metadata_annotations/config.yaml")

def my_pipeline(config):
    # Load the models and transformations.
    models, transformations, classes = load_model_non_hydra(config)
    inputs = config["inputs"]
    outputs = inference(config, models, transformations, inputs)
    # output = outputs['instance_segmentation']["MaskRCNN_ResNet50_FPN"]['paard_persooon.mp4'][0]
    # print([classes[label] for label in output['labels']])

    postprocessed = postprocess_predictions(outputs, config)
    # output = postprocessed['instance_segmentation']["MaskRCNN_ResNet50_FPN"]['paard_persooon.mp4'][0]
    # print([classes[label] for label in output['labels']])

    determine_and_execute_export_function(postprocessed, config)



my_pipeline(config)

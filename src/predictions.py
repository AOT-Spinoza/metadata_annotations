from scripts import keypoints

def inference(config, models, transformations, inputs):
    output_dict = {}
    for model in models:
        for model_name, predictor in model.items():
            if model_name == 'keypoints':
                output_dict['keypoints'] = keypoints.inference(inputs, predictor, transformations['keypoints'], config)
            if model_name == 'semantic_segmentation':
                output_dict['semantic_segmentation'] = []
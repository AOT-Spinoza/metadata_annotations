from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights 
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights

def load_model(config):
    models = {}
    transformations = {}
    for model_name in config.models:
        if model_name == 'keypoints':
            weights = config.models.keypoints.get("weights")
            transformations['keypoints'] = weights.transforms()
            model = keypointrcnn_resnet50_fpn(weights=weights, progress=False)
            model = model.eval()
            models['keypoints'] = model

        if model_name == 'semantic_segmentation':
            weights = config.models.semantic_segmentation.weights
            transformations['semantic_segmentation'] = weights.transforms()
            model =  fcn_resnet101(weights)
            models['semantic_segmentation'] = model

        
    return models, transformations
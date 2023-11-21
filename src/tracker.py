from lib.sort import Sort
import numpy as np
def tracking(predictions, config):
    """
    Applies the SORT algorithm to track instances across frames.

    Args:
        predictions (list of dicts): A list of dictionaries containing the predictions made by the model.

    Returns:
        list of dicts: A list of dictionaries containing the tracked instances.
    """
    # Initialize SORT tracker
    tracker = Sort()
    
    tracked_predictions = []
    
    for prediction in predictions:
    
        bboxes_scores = np.column_stack((prediction['boxes'], prediction['scores']))
        keypoints = prediction.get('keypoints', None)  # Get keypoints if they exist
        masks = prediction.get('masks', None)  # Get masks if they exist
        tracked_bboxes = tracker.update(bboxes_scores)
        tracked_prediction = prediction.copy()
        tracked_prediction['boxes'] = tracked_bboxes[:, :4]
        tracked_prediction['scores'] = prediction['scores'] 
        tracked_prediction['ids'] = tracked_bboxes[:, 4].astype(int)
        if keypoints is not None:
            tracked_prediction['keypoints'] = keypoints  # assuming keypoints are in the same order as boxes
        if masks is not None:
            tracked_prediction['masks'] = masks  # assuming masks are in the same order as boxes
        tracked_predictions.append(tracked_prediction)
    
    return tracked_predictions
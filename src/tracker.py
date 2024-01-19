from library.sort import Sort
import numpy as np
import torch

def tracking(predictions, config):
    """
    Applies the SORT algorithm to track instances across frames.

    Args:
        predictions (list of dicts): A list of dictionaries containing the predictions made by the model.

    Returns:
        list of dicts: A list of dictionaries containing the tracked instances.
    """
    # Initialize SORT tracker
    tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.6)
    print('Tracking')
    tracked_predictions = []
    for prediction in predictions:
        bboxes_scores = np.column_stack((prediction['boxes'], prediction['scores']))
        tracked_bboxes = tracker.update(bboxes_scores)
        new_boxes = tracked_bboxes[:, :4]
        ids = tracked_bboxes[:, 4]
        score_identifier = tracked_bboxes[:, 5]
        # Convert the tensor to a numpy array
        scores_np = prediction['scores'].numpy()
        tracked_prediction = {}
        tracked_prediction['scores'] = torch.from_numpy(score_identifier)
        # Create a mapping from old scores to their indices
        score_to_index = {score: i for i, score in enumerate(scores_np)}
        # For each key in prediction that is not 'boxes' or 'scores'
        for key in prediction:
            if key not in ['boxes', 'scores']:  
                tensor_list = [prediction[key][score_to_index[score.item()]] for score in tracked_prediction['scores']]
                tracked_prediction[key] = torch.stack(tensor_list)
                if len(tensor_list) > 1:
                    tracked_prediction[key] = tracked_prediction[key].squeeze()
        tracked_prediction['boxes'] = torch.from_numpy(new_boxes)
        tracked_prediction['ids'] = torch.from_numpy(ids.astype(int))
        tracked_prediction['scores'] = torch.from_numpy(score_identifier)
        tracked_predictions.append(tracked_prediction)

    return tracked_predictions



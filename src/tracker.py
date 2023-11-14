from lib.sort import Sort

def track_keypoints(predictions):
    """
    Tracks keypoints in a video using the SORT algorithm.

    Args:
        predictions (list): A list of dictionaries, where each dictionary represents a prediction
            for a single frame in the video. Each dictionary should have the following keys:
            - 'boxes': A list of bounding boxes for the keypoints in the frame.
            - 'scores': A list of confidence scores for each bounding box.

    Returns:
        list: A list of dictionaries, where each dictionary represents a prediction for a single
        frame in the video. Each dictionary should have the following keys:
        - 'boxes': A list of bounding boxes for the keypoints in the frame.
        - 'scores': A list of confidence scores for each bounding box.
        - 'id': A unique ID assigned to each tracked object by the SORT algorithm.
    """
    
    # Initialize tracker
    tracker = Sort()

    tracked_predictions = []

    # For each prediction...
    for prediction in predictions:
        # Get bounding boxes and scores
        bboxes = prediction['boxes'].tolist()
        scores = prediction['scores'].tolist()

        # Format for SORT: [x1, y1, x2, y2, score]
        bboxes_scores = [bbox + [score] for bbox, score in zip(bboxes, scores)]

        # Update tracker
        tracked_bboxes = tracker.update(bboxes_scores)

        # Assign tracked IDs to predictions
        for i, bbox in enumerate(tracked_bboxes):
            tracked_prediction = prediction.copy()  # Create a new dictionary
            tracked_prediction['id'] = int(bbox[4])  # The unique ID assigned by SORT
            tracked_predictions.append(tracked_prediction)

    return tracked_predictions
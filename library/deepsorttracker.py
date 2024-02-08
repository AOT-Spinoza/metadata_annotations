import torch
from library.deep_sort_pytorch.utils.parser import get_config
from library.deep_sort_pytorch.deep_sort import DeepSort

class DeepSortTracker:
    def __init__(self, config_path):
        self.cfg = get_config()
        self.cfg.merge_from_file(config_path)
        self.deepsort = DeepSort(
            self.cfg.DEEPSORT.REID_CKPT,
            max_dist=self.cfg.DEEPSORT.MAX_DIST, 
            min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=self.cfg.DEEPSORT.NMS_MAX_OVERLAP, 
            max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=self.cfg.DEEPSORT.MAX_AGE, 
            n_init=self.cfg.DEEPSORT.N_INIT, 
            nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
            use_cuda=True
        )


    def update(self, bbox_xywh, confidences, class_ids, frame):
        # Convert to tensors
        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confidences)
        
        # Update DeepSort with current frame detections
        outputs = self.deepsort.update(xywhs, confss, class_ids, frame)
        # Format tracking results
        tracking_results = []

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            tracking_results = [bbox_xyxy, identities, object_id]

        return tracking_results

from typing import List

from ultralytics import YOLO
import numpy as np
from filterpy.kalman import KalmanFilter
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment
import torch

from src.mltypes import Detection, Track, Box


class Detector:
    def __init__(self, pretrained_model: str, classes: List[str]) -> None:
        # Load a model
        self.model = YOLO(pretrained_model)
        self.model_classes = self.model.names
        self.selected_classes = dict(
            filter(lambda item: item[1] in classes, self.model_classes.items())
        )
        
    def predict(self, img: np.array):
        detections: List[Detection] = []
        # Run batched inference on a list of images
        result = self.model.predict([img], classes=list(self.selected_classes.keys()))[0]

        # Process results list
        boxes = result.boxes  # Boxes object for bounding box outputs
        
        for (bbox, conf) in zip(boxes.xyxy.numpy(), boxes.conf.numpy()):
            x1, y1, x2, y2 = list(map(int, bbox))
            detections.append(Detection(Box(x1, y1, x2 - x1, y2 - y1), conf))
            
        return detections
    

class SortTracker:
    def __init__(self) -> None:
        self.kalman_filters: dict[int, BboxKalmanFilter] = {}
        self.tracks: List[Track] = []

    def __call__(self, detections: List[Detection]):
        if len(self.tracks) > 0:
            detection_bboxes = [det.bbox.xyxy for det in detections]
            tracker_bboxes = [track.bbox.xyxy for track in self.tracks]
            
            iou_values = (
                box_iou(
                    torch.tensor(detection_bboxes, dtype=np.int),
                    torch.tensor(tracker_bboxes, dtype=np.int),
                )
                .cpu()
                .detach()
                .numpy()
            )
            row_idx, col_idx = linear_sum_assignment(iou_values, True)
            
            for i, j in zip(row_idx, col_idx):
                bbox = detections[i].bbox
                track_id = self.tracks[j].id
                
                # Use position predictions from Kalman Filters
                pred_state = self.kalman_filters[track_id].kf.x[:4, 0]
                self.tracks[j].bbox = Box.from_kf_state(pred_state)
                
                # Update the Kalman filter associated to the track
                self.kalman_filters[track_id](bbox)
                
            # mark new detections
            unmatched_rows = [row for row in range(iou_values.shape[0]) if row not in row_idx]
            detections = [detections[i] for i in unmatched_rows]
            
            # mark lost tracks
            unmatched_cols = [col for col in range(iou_values.shape[1]) if col not in col_idx]
            for i in unmatched_cols:
                track_id = self.tracks[i].id
                del self.kalman_filters[track_id]
            
            self.tracks = [track for i, track in enumerate(self.tracks) if i not in unmatched_cols]
            
        # initialize new tracks
        for det in detections:
            bbox = det.bbox
            new_track = Track(
                bbox=bbox,
            )
            self.tracks.append(new_track)
            self.kalman_filters[new_track.id] = BboxKalmanFilter(bbox)


class BboxKalmanFilter:
    def __init__(self, bbox: Box):
        dt = 1
        # Initialize the Kalman Filter object
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # Initial state (position and velocity)
        self.kf.x[:4, 0] = np.array(bbox.kf_state)  # [x1_pos, y1_pos, x2_pos, y2_pos], not chaning the velocity here, keeping it to whatever it's initializaed to

        # State transition matrix F (constant velocity model)
        self.kf.F = np.array([[1, 0, 0, 0, dt, 0, 0], # Ex: x1_pos_new = [x1_pos_old + x_vel * dt]
                        [0, 1, 0, 0, 0, dt, 0],
                        [0, 0, 1, 0, 0, 0, dt],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1]])

        # Measurement function H (we can only observe position)
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0]])

        # Process noise covariance Q (assume some uncertainty in process)
        self.kf.Q = np.eye(7) * 0.1

        # Measurement noise covariance R (assume some uncertainty in measurements)
        self.kf.R = np.eye(4) * 5

        # Covariance matrix P (initial uncertainty)
        self.kf.P = np.eye(7) * 10  # Large initial uncertainty
        
        self(bbox)
        
    def __call__(self, bbox: Box):
        self.kf.predict()
        self.kf.update(bbox.kf_state)

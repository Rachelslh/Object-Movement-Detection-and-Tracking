from typing import List, Dict
from collections import defaultdict

from ultralytics import YOLO
import numpy as np
from filterpy.kalman import KalmanFilter
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment
import torch

from src.mltypes import Detection, Track, Box


class Detector:
    def __init__(
        self, pretrained_model: str, confidence: float, classes: List[str]
    ) -> None:
        # Load a model
        self.model = YOLO(pretrained_model, task="detection")
        self.confidence_threshold = confidence
        self.model_classes = self.model.names
        self.selected_classes = dict(
            filter(lambda item: item[1] in classes, self.model_classes.items())
        )

    def predict(self, img: np.array):
        """
        Performs object detection on the input image and returns a list of detected objects.

        Args:
            img (np.array): Input image as a NumPy array on which detection will be performed.

        Returns:
            List[Detection]: A list of `Detection` objects, each containing:
                            - A `Box` object (x1, y1, width, height) representing the bounding box of the detected object.
                            - A confidence score (float) for the detection.
        """
        detections: List[Detection] = []
        # Run batched inference on a list of images
        result = self.model.predict(
            [img],
            conf=self.confidence_threshold,
            classes=list(self.selected_classes.keys()),
        )[0]

        # Process results list
        boxes = result.boxes  # Boxes object for bounding box outputs

        for (bbox, conf) in zip(boxes.xyxy.numpy(), boxes.conf.numpy()):
            x1, y1, x2, y2 = list(map(int, bbox))
            detections.append(Detection(Box(x1, y1, x2 - x1, y2 - y1), conf))

        return detections


class SortTracker:
    def __init__(self) -> None:
        self.tracks: List[Track] = []
        self.kalman_filters: dict[int, BboxKalmanFilter] = {}
        self.velocity_per_track: dict[int, Dict[int, float]] = defaultdict(dict)

        self.step = 0

    def __call__(self, detections: List[Detection]):
        """
        Updates the tracker state by associating detected objects to existing tracks, predicting future states
        using Kalman Filters, and initializing new tracks if needed.

        Args:
            detections (List[Detection]): A list of `Detection` objects containing the bounding boxes (`Box`)
                                        and confidence scores of detected objects in the current frame.
        """
        if len(detections) > 0 and len(self.tracks) > 0:
            detection_bboxes = [det.bbox.xyxy for det in detections]
            tracker_bboxes = [track.bbox.xyxy for track in self.tracks]

            iou_values = (
                box_iou(
                    torch.tensor(detection_bboxes, dtype=np.int),
                    torch.tensor(tracker_bboxes, dtype=np.int),
                )
                .cpu()
                .numpy()
            )
            row_idx, col_idx = linear_sum_assignment(iou_values, True)

            for i, j in zip(row_idx, col_idx):
                bbox = detections[i].bbox
                track_id = self.tracks[j].id
                pred_state = self.kalman_filters[track_id].kf.x
                current_velocity = np.sqrt(
                    pred_state[4, 0] ** 2 + pred_state[5, 0] ** 2
                )
                if iou_values[i, j] <= 0:
                    self.tracks[j].bbox = Box.from_kf_state(pred_state)
                    self.velocity_per_track[track_id][self.step] = current_velocity
                    continue

                # Use position predictions from Kalman Filters
                self.tracks[j].bbox = Box.from_kf_state(pred_state)
                self.velocity_per_track[track_id][self.step] = current_velocity
                # Update the Kalman filter associated to the track
                self.kalman_filters[track_id](bbox)

            # mark new detections
            unmatched_rows = [
                row for row in range(iou_values.shape[0]) if row not in row_idx
            ]
            detections = [detections[i] for i in unmatched_rows]

            # mark lost tracks
            unmatched_cols = [
                col for col in range(iou_values.shape[1]) if col not in col_idx
            ]
            for i in unmatched_cols:
                track_id = self.tracks[i].id
                del self.kalman_filters[track_id]

            self.tracks = [
                track for i, track in enumerate(self.tracks) if i not in unmatched_cols
            ]

        # initialize new tracks
        for det in detections:
            bbox = det.bbox
            new_track = Track(
                bbox=bbox,
            )
            self.tracks.append(new_track)
            self.kalman_filters[new_track.id] = BboxKalmanFilter(bbox)

        self.step += 1


class BboxKalmanFilter:
    def __init__(self, bbox: Box):
        # Define a time step
        dt = 1
        # Initialize the Kalman Filter object
        self.kf = KalmanFilter(dim_x=6, dim_z=4)

        # Initial state: position [center_x, center_y, area, aspect_ratio], velocity initial values should be zeroed-out by default
        self.kf.x[:4, 0] = np.array(bbox.kf_state)

        # State transition matrix F (constant velocity model)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, dt, 0],  # xc_pos_new = [xc_pos_old + x_vel * dt]
                [0, 1, 0, 0, 0, dt],  # Same for yc here
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        # Measurement function H (we can only retrieve position from model detections)
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]
        )

        self(bbox)

    def __call__(self, bbox: Box):
        """
        Updates the Kalman Filter with the current bounding box state.

        Args:
            bbox (Box): The `Box` object representing the detected object's bounding box in the current frame.
                        The bounding box state (center, area, aspect ratio) is extracted and used to update the filter.
        """
        self.kf.predict()
        self.kf.update(bbox.kf_state)

from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np


# Global counter for unique IDs
_id_counter = 0


def generate_id():
    """
    Generates a unique integer ID by incrementing a global counter.

    This function maintains a global `_id_counter` variable, which is incremented
    each time the function is called. The new value of the counter is returned,
    ensuring a unique ID is generated sequentially.

    Returns:
        int: A unique integer ID.
    """
    global _id_counter
    _id_counter += 1
    return _id_counter


@dataclass
class Box:
    x1: int
    y1: int
    w: int
    h: int

    @property
    def xyxy(self) -> List[int]:
        return self.x1, self.y1, self.x1 + self.w, self.y1 + self.h

    @property
    def center(self) -> Tuple[float, float]:
        return self.x1 + self.w / 2, self.y1 + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        return round(self.w / self.h, 2)

    @property
    def kf_state(self):
        """
        Represents the Kalman Filter state vector for this box, consisting of its center coordinates,
        area, and aspect ratio.

        Returns:
            List[float]: The Kalman Filter state vector [cx, cy, area, aspect_ratio].
        """
        return [*self.center, self.area, self.aspect_ratio]

    @classmethod
    def from_kf_state(cls, state: np.ndarray):
        """
        Reconstructs a Box object from a Kalman Filter state vector. The Kalman filter state is
        assumed to be in the form (cx, cy, area, aspect_ratio, vel_x, vel_y), though only the first
        four elements are used here.

        Args:
            state (np.ndarray): The Kalman filter state vector (cx, cy, area, aspect_ratio, ...).

        Returns:
            Box: A Box object reconstructed from the Kalman Filter state.
        """
        state = state[:4, 0]
        w = int(np.sqrt(state[2] * state[3]))
        h = int(state[2] / w)

        x1 = int(state[0] - w / 2.0)
        y1 = int(state[1] - h / 2.0)

        return cls(x1, y1, w, h)


@dataclass
class Detection:
    bbox: Box
    conf: float


@dataclass
class Track:
    bbox: Box
    id: int = field(default_factory=generate_id)

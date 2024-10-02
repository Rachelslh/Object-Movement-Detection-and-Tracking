from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np


# Global counter for unique IDs
_id_counter = 0


def generate_id():
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
        return [*self.center, self.area, self.aspect_ratio]

    @classmethod
    def from_kf_state(cls, state: np.ndarray):
        """
        kalman filter state 'x' is defined in our use-case as (cx,cy,area,aspect_ratio, vel_x, vel_y)
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

from typing import List, Union, Tuple

from omegaconf import OmegaConf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.model import Detector, SortTracker
from src.mltypes import Detection, Track


def plot(
    image: np.array,
    detections: List[Union[Detection, Track]],
    color: Tuple[int, int, int],
):
    """
    Draws bounding boxes and labels on the image for detections or tracks.

    Args:
        image (np.array): The input image (as a NumPy array) on which to draw the bounding boxes.
        detections (List[Union[Detection, Track]]): A list of `Detection` or `Track` objects.
            - If the object is of type `Detection`, it draws the bounding box and confidence score.
            - If the object is of type `Track`, it draws the bounding box and the track ID.
        color (Tuple[int, int, int]): The color for the bounding boxes, specified as a tuple (B, G, R).
    """

    for det in detections:
        x1, y1, x2, y2 = det.bbox.xyxy
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # Blue box
        if isinstance(det, Detection):
            cv2.putText(
                image,
                f"confidence {det.conf:.2f}",
                (x1, y1 + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        elif isinstance(det, Track):
            cv2.putText(
                image,
                f"Track {det.id}",
                (x1, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )


config = OmegaConf.load("src/config.yaml")
detector = Detector(**config.detection)
tracker = SortTracker()

cap = cv2.VideoCapture(config.inference.input_path)

# Retrieve frame width, height, and FPS
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object using input video's params
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
out = cv2.VideoWriter(
    config.inference.output_video, fourcc, fps, (frame_width, frame_height)
)

if not cap.isOpened():
    print(f"Error: Cannot open video file {config.inference.input_path}")
    exit()

# Loop through the video frame by frame
while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video file or error in reading the video.")
        break

    # Run model inference
    detections = detector.predict(frame)
    tracker(detections)

    # Plot results on the frame
    plot(frame, detections, (255, 0, 0))
    plot(frame, tracker.tracks, (0, 0, 255))

    # Display the frame
    # cv2.imshow("Video", frame)

    # Write the frame to the video
    out.write(frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Plot a graph that shows each track's movement over time
frames = np.arange(n_frames)
for track_id, velocities in tracker.velocity_per_track.items():
    velocity_per_frame = np.zeros((frames.shape[0]))
    velocity_per_frame[list(velocities.keys())] = list(velocities.values())
    # Plot the velocity over time
    plt.plot(frames, velocity_per_frame, label=f"Object {track_id}")

# Customize the plot
plt.title("Objects Movement (Velocity) Over Time")
plt.xlabel("Time (frames)")
plt.ylabel("Velocity (pixels/frame)")
plt.legend()
plt.savefig(config.inference.output_graph)

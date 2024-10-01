from omegaconf import OmegaConf
import numpy as np
import cv2
from typing import List, Union, Tuple

from src.model import Detector, SortTracker
from src.mltypes import Detection, Track


def plot(image: np.array, detections:List[Union[Detection, Track]], color: Tuple[int, int, int]):
    for det in detections:
        x1, y1, x2, y2 = det.bbox.xyxy
        cv2.rectangle(image, (x1 , y1), (x2, y2), color, 2)  # Blue box
        if isinstance(det, Detection):
            cv2.putText(image, f'confidence {det.conf:.2f}', (x1, y1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        elif isinstance(det, Track):
            cv2.putText(image, f'Track {det.id}', (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


config = OmegaConf.load("src/config.yaml")
detector = Detector(**config.detection)
tracker = SortTracker()

# Open the video file
cap = cv2.VideoCapture(config.inference.path)

# Retrieve frame width, height, and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#out = cv2.VideoWriter(config.inference.output_video, fourcc, fps, (frame_width, frame_height))

# Check if the video file was successfully opened
if not cap.isOpened():
    print(f"Error: Cannot open video file {config.inference.path}")
    exit()

# Loop through the video frame by frame
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If frame is read correctly, ret will be True
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
    cv2.imshow('Video', frame)
    
    # Write the frame to the video
    #out.write(frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
# Release the VideoWriter object
#out.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
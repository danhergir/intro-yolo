import argparse
import cv2
from ultralytics import YOLO
import numpy as np
import supervision as sv

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Webcam Example")
    parser.add_argument('--webcam-resolution', default=[1080, 720], nargs=2, type=int, help="Resolution of the webcam")
    return parser.parse_args()

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Load YOLOv8 model
    model = YOLO("yolov8l.pt")

    # Initialize BoxAnnotator
    box_annotator = sv.BoxAnnotator(thickness=2)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        # Run YOLOv8 inference
        results = model(frame)[0]  # Get the first frame's result

        # Extract boxes, confidences, and class IDs
        boxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes in (x1, y1, x2, y2) format
        confidences = results.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        # Create Detections object manually
        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids
        )

        # Annotate frame
        frame = box_annotator.annotate(scene=frame, detections=detections)

        # Display frame
        cv2.imshow("YOLOv8", frame)

        # Stop loop on ESC key
        if cv2.waitKey(30) == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

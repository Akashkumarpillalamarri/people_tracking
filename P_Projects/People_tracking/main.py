import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# Load YOLO model
model = YOLO("yolov8n.pt")  # lightweight YOLOv8

# DeepSORT tracker
tracker = DeepSort(max_age=30)

# Start webcam
cap = cv2.VideoCapture(0)

prev_positions = {}
fps_start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame, stream=True)

    detections = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # class 0 = person
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                detections.append(([x1, y1, x2-x1, y2-y1], conf, "person"))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        # Get previous position
        if track_id in prev_positions:
            px, py = prev_positions[track_id]
            dx, dy = cx - px, cy - py
            speed = (dx**2 + dy**2) ** 0.5

            # Movement classification
            if speed > 15:
                action = "Running"
                color = (0, 0, 255)
            else:
                action = "Walking"
                color = (255, 255, 0)

            # Direction
            if dx > 0:
                direction = "Right"
            elif dx < 0:
                direction = "Left"
            else:
                direction = "Stationary"

            text = f"ID {track_id} | {action} | {direction}"
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        prev_positions[track_id] = (cx, cy)

    # Show FPS
    elapsed_time = time.time() - fps_start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("People Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

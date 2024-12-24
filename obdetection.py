# obDetection.py
import torch
import cv2
import time

def detect_person():
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    cap = cv2.VideoCapture(0)

    # Frame dimensions and confidence threshold
    width, height = 700, 500
    confidence_threshold = 0.5

    detected_frames = 0  # Count of frames with person detection
    total_frames = 0  # Total frames processed
    start_time = time.time()

    # Log person count with timestamps
    with open("person_count_log.txt", "a") as log_file:
        last_logged_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame
            frame = cv2.resize(frame, (width, height))

            # Perform object detection
            results = model(frame)
            labels, confidences, boxes = results.xyxyn[0][:, -1], results.xyxyn[0][:, -2], results.xyxyn[0][:, :-2]

            # Count people detected in the frame
            person_count = 0
            person_detected = False

            for i in range(len(labels)):
                confidence = confidences[i].item()
                if confidence < confidence_threshold:
                    continue

                label = int(labels[i].item())
                if model.names[label] == "person":
                    person_count += 1  # Increment person count
                    person_detected = True

                    # Draw bounding box and display confidence score
                    box = boxes[i].numpy()
                    x1, y1, x2, y2 = int(box[0] * frame.shape[1]), int(box[1] * frame.shape[0]), int(
                        box[2] * frame.shape[1]), int(box[3] * frame.shape[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update frame counts based on detection
            if person_detected:
                detected_frames += 1
            total_frames += 1

            # Display person count on the frame
            cv2.putText(frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Log the person count every second
            current_time = time.time()
            if current_time - last_logged_time >= 1:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}, Person Count: {person_count}\n")
                last_logged_time = current_time

            # Show the frame with detections
            cv2.imshow('YOLOv5 Person Detection', frame)

            # Check if 20 seconds have elapsed
            elapsed_time = time.time() - start_time
            if elapsed_time >= 20:
                break

            # Press 'q' to exit manually
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate detection ratio
    detection_ratio = detected_frames / total_frames if total_frames > 0 else 0
    return detection_ratio >= 0.5  # Return True if detection occurred over 50% of the time

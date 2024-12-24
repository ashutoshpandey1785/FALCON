# main.py
import time
from obdetection import detect_person  # Import the detection function
from facialrecog import run_facial_recognition  # Import the facial recognition function

while True:
    print("Running object detection...")
    person_detected = detect_person()

    # Check if detection ratio met the threshold
    if person_detected:
        print("Person detected consistently. Redirecting to facial recognition...")
        time.sleep(2)  # Brief delay before switching to facial recognition

        # Run the facial recognition module
        run_facial_recognition()
        break  # Exit the loop after running facial recognition
    else:
        print("Person not detected consistently. Continuing object detection...")

    # Allow exit with 'q' keypress (handled inside obDetection.py)

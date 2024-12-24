import sys
import torch
import cv2
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QFrame
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
import time

# Configure loggingpip install PyQt5
logging.basicConfig(
    filename="application.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class VideoThread(QThread):
    image_updated = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = True
        self.model = None
        self.confidence_threshold = 0.5
        self.yolo_active = False
        self.facial_recognition_active = False

        # Load the YOLO model during initialization
        self.load_model()

    def run(self):
        self.cap = cv2.VideoCapture(0)
        logging.info("Video thread started.")
        start_time = None
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Failed to capture frame.")
                break

            # Resize frame for consistency
            frame = cv2.resize(frame, (700, 500))

            # If YOLO is active, process the frame
            if self.yolo_active and self.model is not None:
                frame = self.process_frame(frame)

                # Start facial recognition 10 seconds after detection
                if start_time is None:
                    start_time = time.time()

                if time.time() - start_time >= 10:
                    self.yolo_active = False
                    self.facial_recognition_active = True
                    self.perform_facial_recognition()
                    start_time = None  # Reset the timer after triggering facial recognition

            # Emit the frame to update the UI
            self.image_updated.emit(frame)

    def process_frame(self, frame):
        # Run YOLOv5 model on the frame
        results = self.model(frame)
        detections = results.xyxy[0]  # Format: [x1, y1, x2, y2, confidence, class]

        person_count = 0

        for detection in detections:
            x1, y1, x2, y2, confidence, cls = detection
            if confidence < self.confidence_threshold:
                continue

            if int(cls) == 0:  # Class 0 corresponds to "person" in YOLOv5
                person_count += 1
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Log the person count
        logging.info(f"Person Count: {person_count}")

        # Display person count on the frame
        cv2.putText(frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def perform_facial_recognition(self):
        # Facial recognition setup
        logging.info("Starting facial recognition...")
        video = cv2.VideoCapture(0)
        video.set(cv2.CAP_PROP_FPS, 30)
        facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("Trainer.yml")
        name_list = ["", "Harshit Sharma", "Gaurang Dwivedi", "Kavya Chaurasia", "Ashutosh Pandey"]

        while self.facial_recognition_active:
            ret, frame = video.read()
            if not ret:
                logging.warning("Failed to capture frame for facial recognition.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                serial, conf = recognizer.predict(gray[y:y + h, x:x + w])

                # Display recognized name if confidence is above threshold
                if conf < 60:
                    name = name_list[serial] if 0 <= serial < len(name_list) else "Unknown"
                else:
                    name = "Unknown"

                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Emit the frame to update the UI
            self.image_updated.emit(frame)

            # Press 'q' to quit facial recognition mode
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.facial_recognition_active = False

        video.release()
        cv2.destroyAllWindows()

    def load_model(self):
        # Load the YOLO model
        logging.info("Loading YOLO model...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        logging.info("YOLO model loaded.")

    def activate_yolo(self, active):
        self.yolo_active = active
        if active:
            logging.info("YOLO activated.")
        else:
            logging.info("YOLO deactivated.")

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        logging.info("Video thread stopped.")
        self.wait()  # Wait for the thread to finish


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Person Detection and Facial Recognition")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF;")  # Dark theme
        self.setup_ui()
        logging.info("Application started.")

    def setup_ui(self):
        # Central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        # Video display label
        self.label = QLabel("Video Feed", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFrameStyle(QFrame.Box)
        self.label.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF; font-size: 16px; border: 1px solid #555555;")
        self.main_layout.addWidget(self.label)

        # Button layout
        button_layout = QHBoxLayout()

        # Start YOLO button
        self.start_yolo_button = QPushButton("Start YOLO")
        self.start_yolo_button.clicked.connect(self.start_yolo)
        self.start_yolo_button.setStyleSheet("background-color: #4CAF50; color: #FFFFFF; font-size: 14px;")
        button_layout.addWidget(self.start_yolo_button)

        # Stop YOLO button
        self.stop_yolo_button = QPushButton("Stop YOLO")
        self.stop_yolo_button.clicked.connect(self.stop_yolo)
        self.stop_yolo_button.setStyleSheet("background-color: #F44336; color: #FFFFFF; font-size: 14px;")
        button_layout.addWidget(self.stop_yolo_button)

        # Start Facial Recognition button
        self.start_facial_recognition_button = QPushButton("Start Facial Recognition")
        self.start_facial_recognition_button.clicked.connect(self.start_facial_recognition)
        self.start_facial_recognition_button.setStyleSheet("background-color: #2196F3; color: #FFFFFF; font-size: 14px;")
        button_layout.addWidget(self.start_facial_recognition_button)

        # Stop Facial Recognition button
        self.stop_facial_recognition_button = QPushButton("Stop Facial Recognition")
        self.stop_facial_recognition_button.clicked.connect(self.stop_facial_recognition)
        self.stop_facial_recognition_button.setStyleSheet("background-color: #FFC107; color: #000000; font-size: 14px;")
        button_layout.addWidget(self.stop_facial_recognition_button)

        # Exit button
        self.exit_button = QPushButton("Completely Exit")
        self.exit_button.clicked.connect(self.close_application)
        self.exit_button.setStyleSheet("background-color: #000000; color: #FFFFFF; font-size: 14px;")
        button_layout.addWidget(self.exit_button)

        # Add button layout to the main layout
        self.main_layout.addLayout(button_layout)

        # Set the layout for the central widget
        self.central_widget.setLayout(self.main_layout)

        # Video thread setup
        self.video_thread = VideoThread()
        self.video_thread.image_updated.connect(self.update_image)
        self.video_thread.start()

    def update_image(self, frame):
        # Convert frame to QImage and display on label
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)

    def start_yolo(self):
        self.video_thread.activate_yolo(True)

    def stop_yolo(self):
        self.video_thread.activate_yolo(False)

    def start_facial_recognition(self):
        self.video_thread.facial_recognition_active = True

    def stop_facial_recognition(self):
        self.video_thread.facial_recognition_active = False

    def close_application(self):
        self.video_thread.stop()
        QApplication.quit()
        logging.info("Application closed.")
        sys.exit()
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

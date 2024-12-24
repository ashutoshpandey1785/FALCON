# testmodel.py
import cv2

def run_facial_recognition():
    # Facial recognition setup
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FPS, 30)
    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")
    name_list = ["", "Harshit Sharma","Gaurang Dwivedi", "Kavya Chaurasia", "Ashutosh Pandey"]

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            serial, conf = recognizer.predict(gray[y:y+h, x:x+w])

            # Display recognized name if confidence is above threshold
            if conf < 60:
                name = name_list[serial] if 0 <= serial < len(name_list) else "Unknown"
            else:
                name = "Unknown"

            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("Facial Recognition", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()

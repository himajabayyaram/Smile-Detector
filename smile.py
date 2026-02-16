import cv2
import serial
import time

# -----------------------------
# 1. Connect to Arduino
# -----------------------------
# 🔴 IMPORTANT: change COM port to match your Arduino
# Example: 'COM3', 'COM4', 'COM5', etc.
arduino = serial.Serial('COM8', 9600, timeout=1)
time.sleep(2)  # wait for Arduino to reset

# -----------------------------
# 2. Load Haar Cascade models
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

# -----------------------------
# 3. Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

last_state = '0'  # remember last state sent to Arduino

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    smile_detected = False

    for (x, y, w, h) in faces:
        # Draw face rectangle (for display only)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Region of interest (face only)
        face_roi_gray = gray[y:y + h, x:x + w]
        face_roi_color = frame[y:y + h, x:x + w]

        # Detect smiles inside the face
        smiles = smile_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.8,
            minNeighbors=20
        )

        if len(smiles) > 0:
            smile_detected = True
            # Draw smile rectangle (first one)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(face_roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
                break  # just draw one

    # -----------------------------
    # 4. Send data to Arduino
    # -----------------------------
    if smile_detected and last_state != '1':
        arduino.write(b'1')       # turn relay/LED ON
        last_state = '1'
        print("Smile detected → ON")
    elif not smile_detected and last_state != '0':
        arduino.write(b'0')       # turn relay/LED OFF
        last_state = '0'
        print("No smile → OFF")

    # -----------------------------
    # 5. Show video window
    # -----------------------------
    cv2.imshow("Smile Detector", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# 6. Cleanup
# -----------------------------
arduino.write(b'0')  # make sure it's OFF at the end
cap.release()
cv2.destroyAllWindows()
arduino.close()

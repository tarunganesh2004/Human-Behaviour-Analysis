import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load Facial Emotion Recognition Model
emotion_model = load_model("model.h5")  # Update with your model path
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Initialize MediaPipe Pose for Body Language
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Load Haarcascade for Face Detection (For Emotion Model)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Function to classify posture based on body landmarks
def classify_posture(landmarks):
    if not landmarks:
        return "Unknown"

    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y

    if left_wrist < left_shoulder and right_wrist < right_shoulder:
        return "Hands Up"
    elif left_wrist < left_shoulder or right_wrist < right_shoulder:
        return "Pointing Up"
    elif left_hip > left_knee and right_hip > right_knee:
        return "Sitting"
    elif left_hip < left_knee and right_hip < right_knee:
        return "Standing"
    else:
        return "Unknown"


# Function to determine behavior based on emotion and posture
def determine_behavior(emotion, posture):
    behavior_mapping = {
        ("Happy", "Hands Up"): "Excited/Celebrating",
        ("Sad", "Sitting"): "Depressed/Tired",
        ("Angry", "Pointing Up"): "Argumentative/Aggressive",
        ("Neutral", "Looking Down"): "Distracted/Uninterested",
        ("Surprise", "Standing"): "Alert/Aware",
        ("Fear", "Hands Up"): "Scared/Defensive",
    }
    return behavior_mapping.get((emotion, posture), "Uncertain Behavior")


# Load test image
image_path = "test/image1.png"  # Update with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize window to fit text properly
cv2.namedWindow("Human Behavior Analysis", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Human Behavior Analysis", 800, 600)

# Detect pose landmarks
pose_results = pose.process(image_rgb)

# Detect face for emotion analysis
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)

emotion_detected = "Unknown"
for x, y, w, h in faces:
    roi_gray = gray[y : y + h, x : x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi_gray = roi_gray.astype("float") / 255.0
    roi_gray = img_to_array(roi_gray)
    roi_gray = np.expand_dims(roi_gray, axis=0)

    predictions = emotion_model.predict(roi_gray)[0]
    emotion_detected = emotion_labels[np.argmax(predictions)]

# Get posture classification
posture_detected = (
    classify_posture(pose_results.pose_landmarks.landmark)
    if pose_results.pose_landmarks
    else "Unknown"
)

# Determine final behavior
final_behavior = determine_behavior(emotion_detected, posture_detected)

# Print results in the terminal
print(f"Emotion Detected: {emotion_detected}")
print(f"Posture Detected: {posture_detected}")
print(f"Final Behavior Analysis: {final_behavior}")

# Determine image size dynamically
image_height, image_width, _ = image.shape

# Dynamically adjust font scale based on image size
font_scale = min(image_width, image_height) / 700  # Adjust as needed
thickness = int(font_scale * 3)  # Scale thickness too

# Define text positions
x_start = int(image_width * 0.05)  # Left margin
y_start = int(image_height * 0.15)  # Top margin
line_gap = int(image_height * 0.08)  # Spacing between lines

# Put text on the image
cv2.putText(
    image,
    f"Emotion: {emotion_detected}",
    (x_start, y_start),
    cv2.FONT_HERSHEY_SIMPLEX,
    font_scale,
    (0, 255, 0),
    thickness,
)
cv2.putText(
    image,
    f"Posture: {posture_detected}",
    (x_start, y_start + line_gap),
    cv2.FONT_HERSHEY_SIMPLEX,
    font_scale,
    (0, 255, 0),
    thickness,
)
cv2.putText(
    image,
    f"Behavior: {final_behavior}",
    (x_start, y_start + 2 * line_gap),
    cv2.FONT_HERSHEY_SIMPLEX,
    font_scale,
    (0, 0, 255),
    thickness,
)

# Show Image
cv2.imshow("Human Behavior Analysis", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

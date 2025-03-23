import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

# Load Image
image_path = "your_image.jpg"  # Change this to your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process Image
results = pose.process(image_rgb)

# Check if pose detected
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark

    # Extract key points
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Calculate midpoints
    shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_avg_y = (left_hip.y + right_hip.y) / 2
    knee_avg_y = (left_knee.y + right_knee.y) / 2

    # Rule-based classification
    if abs(shoulder_avg_y - hip_avg_y) > 0.2 and abs(hip_avg_y - knee_avg_y) > 0.2:
        posture = "Standing"
    elif abs(shoulder_avg_y - hip_avg_y) < 0.2 and abs(hip_avg_y - knee_avg_y) < 0.2:
        posture = "Sitting"
    elif left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y:
        posture = "Hand Raised"
    else:
        posture = "Unknown"

    # Draw pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show classification result
    cv2.putText(
        image,
        f"Posture: {posture}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

# Display output
cv2.imshow("Pose Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
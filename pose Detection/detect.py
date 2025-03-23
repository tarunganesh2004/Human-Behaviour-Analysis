import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Function to classify posture
def classify_posture(landmarks):
    if not landmarks:
        return "Unknown"

    # Extract key landmarks
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y

    # Define rules
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


# Load an image
image_path = "test/image.png"  
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process image with MediaPipe
results = pose.process(image_rgb)

if results.pose_landmarks:
    posture = classify_posture(results.pose_landmarks.landmark)
    print("Detected Posture:", posture)
else:
    print("No person detected!")
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


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
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y

    # Define rules
    if left_wrist < left_shoulder and right_wrist < right_shoulder:
        return "Hands Up (Both Hands Raised)"
    elif left_wrist < left_shoulder or right_wrist < right_shoulder:
        return "Pointing Up (One Hand Raised)"
    elif left_elbow < left_shoulder and right_elbow < right_shoulder:
        return "Arms Crossed"
    elif left_hip > left_knee and right_hip > right_knee:
        return "Sitting"
    elif left_hip < left_knee and right_hip < right_knee:
        return "Standing"
    elif left_hip < right_hip - 0.05:  # Slightly leaning left
        return "Leaning Left"
    elif right_hip < left_hip - 0.05:  # Slightly leaning right
        return "Leaning Right"
    else:
        return "Unknown"


# Load an image
image_path = "test/image3.png"  # Change this to your image path
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found!")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process image with MediaPipe
results = pose.process(image_rgb)

if results.pose_landmarks:
    posture = classify_posture(results.pose_landmarks.landmark)
    print("Detected Posture:", posture)

    # Draw landmarks on image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the detected posture on the image
    cv2.putText(
        image,
        f"Posture: {posture}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

# Show the output
cv2.imshow("Pose Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
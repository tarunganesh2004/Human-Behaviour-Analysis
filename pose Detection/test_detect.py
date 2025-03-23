import cv2
import mediapipe as mp

# Initialize Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load image
image_path = "image_copy.png"  # Change to your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect pose
results = pose.process(image_rgb)

# Draw landmarks
if results.pose_landmarks:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Show result
cv2.imshow("Pose Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

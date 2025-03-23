## Posture Classification

Now that we can detect the pose, we need to analyze and classify postures.
For this, we will:

Extract key points (landmarks like shoulders, elbows, knees, etc.).

Define posture categories (e.g., standing, sitting, raising hand).

Train a model to classify postures.

## Approach 1: Rule-based Classification
(No ML Required)
We define rules based on angles and key points:

Standing → Shoulder & hip alignment, straight legs.

Sitting → Knee at ~90° angle, lower hips.

Hand Raised → One arm above head.

✅ Quick & simple but not highly flexible.
import cv2
import mediapipe as mp
import pickle
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Load your image
image = cv2.imread("0-to-9.jpg")
if image is None:
    raise FileNotFoundError("Make sure 0-to-9.jpg is in the same folder as this script!")

# Convert image to RGB (MediaPipe expects RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process with MediaPipe Hands
result = hands.process(image_rgb)

dataset = {"dataset": [], "labels": []}

if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
        # Extract all (x, y) coordinates
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)

        # You can assign a dummy label for now (e.g., 0)
        dataset["dataset"].append(landmarks)
        dataset["labels"].append(0)

    # Save as pickle
    with open("extracted_landmarks.pickle", "wb") as f:
        pickle.dump(dataset, f)

    print("✅ Extracted landmarks saved to extracted_landmarks.pickle")
else:
    print("❌ No hands detected in the image. Try a clearer image or multiple gesture images.")

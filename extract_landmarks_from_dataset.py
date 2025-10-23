import mediapipe as mp
import cv2
import os
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.1)
mp_draw = mp.solutions.drawing_utils

data = []

# Go through dataset/0, dataset/1, ..., dataset/9
for label in range(10):
    folder_path = f'dataset/{label}'
    if not os.path.exists(folder_path):
        continue

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmark_coordinates = hand_landmarks.landmark
                x_coordinates = [coordinates.x for coordinates in landmark_coordinates]
                y_coordinates = [coordinates.y for coordinates in landmark_coordinates]
                min_x, min_y = min(x_coordinates), min(y_coordinates)
                landmarks = []
                for coordinates in landmark_coordinates:
                    normalized_x = coordinates.x - min_x
                    normalized_y = coordinates.y - min_y
                    landmarks.extend([normalized_x, normalized_y, coordinates.z])
                data.append([landmarks, label])
        else:
            print(f"No hands detected in {file}")

with open('extracted_landmarks.pickle', 'wb') as f:
    pickle.dump(data, f)

print(f"Landmark data saved successfully! Total samples: {len(data)}")

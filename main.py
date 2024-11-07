import pickle 
import cv2
import mediapipe as mp 
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# Import the necessary modules from the Mediapipe library for hand tracking
mp_hands = mp.solutions.hands  # Mediapipe's hand tracking solution
mp_drawing = mp.solutions.drawing_utils  # Utility for drawing landmarks and connections on images
mp_drawing_styles = mp.solutions.drawing_styles  # Provides pre-defined drawing styles for landmarks

# Initialize the Hands model from Mediapipe
hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for video streams; detection happens only in the first frame
    max_num_hands=1,  # Track at most one hand
    min_detection_confidence=0.9  # Minimum confidence score for hand detection
)

labels = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9"}

with open("./rf_model.p", "rb") as f:
    model = pickle.load(f)


rf_model = model["model"]


# Start capturing video from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Start an infinite loop to process each video frame in real-time
while True:
    
    # Read the next frame from the video capture
    ret, frame = cap.read()

    # Lists to store normalized landmark coordinates and x/y coordinates
    normalized_landmarks = []  # To store normalized coordinates
    x_coordinates, y_coordinates = [], []  # To store the x and y coordinates of landmarks

    # Capture another frame (redundant call, you might only need one)
    ret, frame = cap.read()

    # Get the dimensions of the frame (height, width, and color channels)
    height, width, _ = frame.shape

    # Convert the frame from BGR (used by OpenCV) to RGB (used by Mediapipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the Mediapipe Hands model to detect hands
    processed_image = hands.process(frame_rgb)
    
    # Get hand landmarks (if any are detected) from the processed image
    hand_landmarks = processed_image.multi_hand_landmarks

    # If hand landmarks are detected in the frame
    if hand_landmarks:
        # Loop through the detected hand landmarks
        for hand_landmark in hand_landmarks:
            # Draw the hand landmarks and connections on the frame using the predefined styles
            mp_drawing.draw_landmarks(
                frame,  # The original frame
                hand_landmark,  # Detected landmarks for the hand
                mp_hands.HAND_CONNECTIONS,  # Hand connections to be drawn
                mp_drawing_styles.get_default_hand_landmarks_style(),  # Style for hand landmarks
                mp_drawing_styles.get_default_hand_connections_style()  # Style for hand connections
            )

            # Loop through the landmarks of the hand and extract coordinates
            for hand_landmark in hand_landmarks:
                landmark_coordinates = hand_landmark.landmark

                # Store x and y coordinates of the landmarks
                for coordinates in landmark_coordinates:
                    x_coordinates.append(coordinates.x)  # Append x coordinates (normalized 0-1)
                    y_coordinates.append(coordinates.y)  # Append y coordinates (normalized 0-1)

                # Find the minimum x and y values (to be used for normalization)
                min_x, min_y = min(x_coordinates), min(y_coordinates)

                # Normalize the x and y coordinates based on the minimum values
                for coordinates in landmark_coordinates:
                    normalized_x = coordinates.x - min_x  # Normalize x
                    normalized_y = coordinates.y - min_y  # Normalize y
                    normalized_landmarks.extend((normalized_x, normalized_y))  # Store normalized values

        # Convert normalized coordinates to pixel values for bounding box display
        x1 = int(min(x_coordinates) * width)  # Minimum x coordinate scaled to the frame width
        y1 = int(min(y_coordinates) * height)  # Minimum y coordinate scaled to the frame height
        x2 = int(max(x_coordinates) * width)  # Maximum x coordinate scaled to the frame width
        y2 = int(max(y_coordinates) * height)  # Maximum y coordinate scaled to the frame height

        # Prepare the normalized landmarks to be used for model prediction
        sample = np.asarray(normalized_landmarks).reshape(1, -1)  # Reshape the landmarks into a sample
        pred = rf_model.predict(sample)  # Use a pre-trained random forest model to make predictions

        # Get the predicted character/label (from a pre-defined labels list) based on model output
        predicted_character = labels[int(pred[0])]

        # Draw a rectangle around the detected hand based on the bounding box
        cv2.rectangle(frame, (x1 + 10, y1 + 10), (x2, y2), (100, 200, 100), 4)  # Green rectangle

        # Display the predicted character as text on the frame
        cv2.putText(img=frame,                          # Image/frame on which to put text
                    text=predicted_character,           # Text to display (predicted character)
                    org=(x1, y1),                       # Text position (top-left corner of the bounding box)
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                    fontScale=2,                        # Font scale (size)
                    color=(0, 0, 0),                    # Text color (black)
                    thickness=3,                        # Thickness of the text
                    lineType=cv2.LINE_AA)               # Anti-aliased line for smooth text rendering

    # Display the video frame with landmarks, bounding box, and predicted character in a window
    cv2.imshow("Video Mode", frame)

    # Exit the loop if the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture when the loop ends and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()


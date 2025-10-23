# Real-Time-Gesture-Based-Number-Recognition
This project demonstrates a real-time hand gesture recognition system that can detect and classify sign language numbers (0–9) using your webcam. It combines MediaPipe for hand pose detection, OpenCV for video processing, and a Random Forest Classifier for gesture prediction.

# Overview
The system captures live video from your webcam, detects hand landmarks using MediaPipe, and then predicts the number shown using a trained Random Forest model. It highlights the detected hand, displays landmarks, and overlays the recognized number — all in real-time.

# How it works?
1. Data Collection (collect_dataset.py)

Captures images for each gesture (numbers 0–9) using your webcam.
Stores them in separate folders (./dataset/0, ./dataset/1, ...).
You can configure the number of samples and capture conditions.

2. Landmark Extraction (extract_landmarks.py)

Uses MediaPipe Hands to detect 21 key landmarks for each hand.
Extracts x, y coordinates (excluding z for simplicity).
Saves processed data and labels in a pickle file (extracted_landmarks.pickle).

3. Model Training (train_classifier.py)

Trains a Random Forest Classifier on the extracted hand landmark data.
Achieves ~99% accuracy on test samples.
Saves the trained model as rf_model.p for later use.

4. Real-Time Prediction (main.py)

Activates your webcam feed.
Uses the trained model to predict the number you’re showing.
Displays the prediction and bounding box live on-screen.

# Usage
To use the system, simply run the main.py script. It captures video from your webcam, uses MediaPipe for hand pose estimation, and feeds the extracted landmarks to the trained Random Forest model. The model then predicts the number based on the hand gesture, displaying the result within a box around the hand.

The system provides real-time number predictions (0-9), updating with each frame based on the hand gesture and its position.

# Tech Stack
1. Python: Core programming language
2. OpenCV: Capturing and displaying video frames
3. MediaPipe: Detecting hand landmarks
4. Scikit-learn: Training the Random Forest model
5. NumPy: Numerical data processing
6. Pickle: Storing and loading datasets & trained models

---

### Credits
Project idea inspired by (https://github.com/21f3000413/Real-Time-Gesture-Based-Number-Recognition) by Aryan (GitHub username: 21f3000413)

**Developed by **Vasundhara Shivankar** 💫  
> "Empowering interaction through vision and machine learning."

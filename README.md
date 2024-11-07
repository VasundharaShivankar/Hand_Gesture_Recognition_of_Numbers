# Real-Time-Gesture-Based-Number-Recognition
This project showcases the real-time detection of sign language numbers (0-9) through hand gestures. The system uses a webcam to capture hand movements and leverages MediaPipe for hand pose estimation. A trained Random Forest machine learning model then analyzes the positions of hand landmarks to predict the corresponding number accurately

#Dataset
I gathered the dataset using my webcam, generating 1,000 images for each number (0-9) via the collect_images module. These images were then processed with the create_dataset module to extract the x and y coordinates from hand pose landmarks. Each array of landmarks was labeled with the corresponding number and saved as a pickle file.

A key challenge in building the dataset was ensuring accurate predictions across varying hand orientations, distances from the camera, and frame positions. To address this, I captured images in diverse conditions. However, the model’s accuracy could still improve with additional data, especially from different distances and angles.

The complete dataset is about 6GB in size, so it’s not uploaded here, but you can create your own dataset using the modules mentioned.

#Model
The model used for predictions is a Random Forest classifier, achieving over 99% accuracy. I trained the model using the train_classifier module, which used the extracted landmark coordinates. The trained model is saved as a .p file for future use.

#Usage
To use the system, simply run the main.py script. It captures video from your webcam, uses MediaPipe for hand pose estimation, and feeds the extracted landmarks to the trained Random Forest model. The model then predicts the number based on the hand gesture, displaying the result within a box around the hand.

The system provides real-time number predictions (0-9), updating with each frame based on the hand gesture and its position.

import pickle 
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

with open("./extracted_landmarks.pickle", "rb") as f:
    dataset = pickle.load(f)

# Iterate through each item in the dataset's "dataset" list
for i in dataset["dataset"]:
    # Check if the length of the current item is not equal to 42 (expected length for hand landmarks)
    if len(i) != 42:
        # Find the index of the current item in the dataset
        index = dataset["dataset"].index(i)
        
        # Remove the item from both the "dataset" and "labels" lists at the found index
        dataset["dataset"].pop(index)
        dataset["labels"].pop(index)


data = np.asarray(dataset["dataset"])
labels = np.asarray(dataset["labels"])

print(len(data))

labels

# Split the data into training and testing sets
# - data: The features (input data) to be split
# - labels: The corresponding labels (output data) to be split
# - test_size=0.2: 20% of the data will be used for testing, and 80% for training
# - shuffle=True: Shuffle the data before splitting to ensure randomness
# - stratify=labels: Ensure that the split preserves the proportion of each class label in both the training and test sets
# - random_state=42: Set a seed for reproducibility of the results
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)


# Reshape X_test[0] to have shape (1, n_features)
y_pred = model.predict(X_test)
y_pred

score = accuracy_score(y_pred, y_test)

print(f'{score * 100}% of samples were classified correctly !')

with open("./rf_model.p", "wb") as f:
    pickle.dump({"model":model}, f)


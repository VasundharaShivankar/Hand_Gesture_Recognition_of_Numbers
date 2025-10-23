import pickle 
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

with open("./extracted_landmarks.pickle", "rb") as f:
    dataset = pickle.load(f)

# Separate data and labels from the dictionary
data = dataset["dataset"]
labels = dataset["labels"]

# Filter out entries where landmarks length is not 42 (21 landmarks * 2 coordinates, since z is not included in create_dataset.py)
cleaned_data = []
cleaned_labels = []
for d, l in zip(data, labels):
    if len(d) == 42:
        cleaned_data.append(d)
        cleaned_labels.append(l)

data = np.asarray(cleaned_data)
labels = np.asarray(cleaned_labels)

print(len(data))

labels

# Split the data into training and testing sets
# - data: The features (input data) to be split
# - labels: The corresponding labels (output data) to be split
# - test_size=0.2: 20% of the data will be used for testing, and 80% for training
# - shuffle=True: Shuffle the data before splitting to ensure randomness
# - random_state=42: Set a seed for reproducibility of the results
# Note: Removed stratify due to insufficient samples per class
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)


# Reshape X_test[0] to have shape (1, n_features)
y_pred = model.predict(X_test)
y_pred

score = accuracy_score(y_pred, y_test)

print(f'{score * 100}% of samples were classified correctly !')

with open("./rf_model.p", "wb") as f:
    pickle.dump({"model":model}, f)


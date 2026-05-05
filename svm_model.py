# =========================================
# 🐱🐶 Improved SVM with HOG Features
# =========================================

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog

data_dir = "dataset"
categories = ["cat", "dog"]

data = []
labels = []

# Load images
for category in categories:
    path = os.path.join(data_dir, category)
    label = categories.index(category)

    for img in os.listdir(path)[:300]:  # limit
        img_path = os.path.join(path, img)

        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.resize(image, (128, 128))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # HOG feature extraction
        features = hog(gray, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)

        data.append(features)
        labels.append(label)

X = np.array(data)
y = np.array(labels)

print("Total samples:", len(X))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM (RBF kernel)
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

import joblib

# Save model
joblib.dump(model, "svm_model.pkl")

print("Model saved as svm_model.pkl")
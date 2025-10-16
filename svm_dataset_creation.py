"""
SVM Dataset Creation and Model Training Script

This script handles the creation of datasets for emotion detection using SVM (Support Vector Machine).
It processes images from the CK+ dataset, extracts HOG (Histogram of Oriented Gradients) features,
trains an SVM model, and evaluates its performance using various metrics.

The script includes:
- Feature extraction from emotion-labeled images
- Dataset splitting for training and testing
- SVM model training with RBF kernel
- Performance evaluation using hold-out and cross-validation methods
- Model persistence for later use

Author: Emotion Detection Team
"""

import os
import pickle
import random
from collections import defaultdict

import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

from util import gray_image, read_image, resize_image

example_image_master_path = "ckplus/CK+48"


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    """
    Print detailed performance metrics for a trained classifier.
    
    This function evaluates and displays comprehensive performance metrics including
    accuracy, classification report, and confusion matrix for either training or
    testing data.
    
    Args:
        clf: The trained classifier (SVM model).
        X_train (list): Training feature vectors.
        y_train (list): Training labels.
        X_test (list): Testing feature vectors.
        y_test (list): Testing labels.
        train (bool): If True, evaluate on training data; if False, evaluate on test data.
        
    Returns:
        None: This function prints results to console.
    """
    if train:
        pred = clf.predict(X_train)
        clf_report = classification_report(y_train, pred)
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")

    elif train == False:
        pred = clf.predict(X_test)
        clf_report = classification_report(y_test, pred)
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# Feature extraction section (commented out - features already extracted)
# This section would extract HOG features from all images in the CK+ dataset
# and save them to a pickle file for later use in model training.

# image_paths = [
#     os.path.join(path, name)
#     for path, subdirs, files in os.walk(example_image_master_path)
#     for name in files
# ]
# print(f"number of image paths: {len(image_paths)}")

# #test_image_paths = random.sample(image_paths, 100)
# #print(f"test image paths: {test_image_paths})")

# dataset=defaultdict(list)
# for image_path in tqdm(image_paths,total=len(image_paths),desc="converting test images to features"):
#     img = read_image(image_path)
#     img = gray_image(img)
#     img = resize_image(img, 64)
#     feature = hog(img,
#                   orientations=7,
#                   pixels_per_cell=(8, 8),
#                   cells_per_block=(4, 4),
#                   block_norm='L2-Hys',
#                   transform_sqrt=False)

#     label = image_path.split(os.path.sep)[-2]
#     dataset[label].append((image_path,feature))
# # save the dataset to a pickle file
# with open('svm_features.pkl', 'wb') as f:
#     pickle.dump(dataset, f)

# Dataset loading and splitting section
# Load pre-extracted features from pickle file and split into training/testing sets
with open("svm_features.pkl", "rb") as f:
    features_dataset = pickle.load(f)

# Configuration parameters for model training and evaluation
test_size = 0.2  # 20% of data for testing
random_state = 42  # For reproducible results
epochs = 1000  # Maximum iterations for SVM training

# Initialize lists for training and testing data
X_train = []
X_test = []
y_train = []
y_test = []
total_features = []
total_labels = []

# Split dataset into training and testing sets for each emotion class
for i, (label, features_list) in enumerate(features_dataset.items()):
    features = []
    labels = []
    for feature_path, feature in features_list:
        features.append(feature)
        labels.append(label)
        total_features.append(feature)
        total_labels.append(label)
    X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    X_train.extend(X_train_subset)
    X_test.extend(X_test_subset)
    y_train.extend(y_train_subset)
    y_test.extend(y_test_subset)


# Display dataset statistics
print(f"numbers of X_train: {len(X_train)}")
print(f"numbers of X_test: {len(X_test)}")
print(f"numbers of y_train: {len(y_train)}")
print(f"numbers of y_test: {len(y_test)}\n")

# SVM Model Training Section
# Initialize SVM classifier with RBF kernel and optimized parameters
model = SVC(
    kernel="rbf",  # Radial Basis Function kernel for non-linear classification
    gamma="scale",  # Automatic gamma scaling
    C=10,  # Regularization parameter
    random_state=random_state,  # For reproducible results
    max_iter=epochs,  # Maximum training iterations
)

# Train the SVM model on the training data
model.fit(X_train, y_train)

# Model Evaluation Section
# Evaluate model performance using hold-out method
print_score(model, X_train, y_train, X_test, y_test, train=True)
print_score(model, X_train, y_train, X_test, y_test, train=False)

# Cross-validation evaluation for more robust performance assessment
# 3-Fold cross-validation to get mean accuracy and standard deviation
cv = KFold(n_splits=3, random_state=1, shuffle=True)
# Evaluate model using cross-validation
scores = cross_val_score(model, total_features, total_labels, scoring="accuracy", cv=cv, n_jobs=-1)
# Report cross-validation performance

print("SVM MEAN  Accuracy: ", str(np.mean(scores) * 100)[:5] + "%")
print("Standard deviation: ", str(np.std(scores) * 100)[:5] + "%")

# Model Persistence Section
# Save the trained model to disk for later use in emotion detection
with open("svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

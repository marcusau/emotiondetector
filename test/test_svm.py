import os
import pickle
import sys
from pathlib import Path

sys.path.append(os.getcwd())

from predictor import FaceDetector, EmotionPredictor

image_folder_path = "images"
face_image_folder_path = "face_images"
model_path = "svm_model.pkl"

image_filename = "image1.jpg"
image_path = os.path.join(image_folder_path, image_filename)
image_name = Path(image_path).stem

emotion_predictor = EmotionPredictor(model_path)
detector = FaceDetector(image_path)
features = detector.feature_extraction()

print("Detecting faces...")

face_predictions = emotion_predictor.predict(features)
print(face_predictions)

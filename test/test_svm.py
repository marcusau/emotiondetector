import os
from pathlib import Path

from predictor import BatchFeatureExtractor, EmotionPredictor
from util import read_image

image_folder_path = "images"
face_image_folder_path = "face_images"
model_path = "svm_model.pkl"

image_filename = "image1.jpg"
image_path = os.path.join(image_folder_path, image_filename)
image_name = Path(image_path).stem
image = read_image(image_path)

emotion_predictor = EmotionPredictor(model_path)
feature_extractor = BatchFeatureExtractor()

features = feature_extractor.process_batch(image)
print(features)
print("Detecting faces...")

face_predictions = emotion_predictor.predict(features)
print(face_predictions)

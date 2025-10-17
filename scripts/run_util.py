import os
import sys
from pathlib import Path
import dlib

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from util import read_image, gray_image, resize_image, draw_rect, check_ndim
from predictor import FaceDetector, ImageChopper, FeatureExtractor, EmotionPredictor

face_detector = FaceDetector()
chopper = ImageChopper()
extractor = FeatureExtractor()
detector = EmotionPredictor("svm_model.pkl")

image_path = "images/image1.jpg"

image = read_image(image_path)

features = extractor.extract_features(image)
print(features)
result = detector.predict([features])

print(type(result))
for r in result:
    print(type(r))
    

# gray_image = gray_image(image)
# # print(type(gray_image))
# # print(gray_image.ndim)
# # print(gray_image)
# resize_image = resize_image(gray_image,500)
# print(type(resize_image))
# print(resize_image.ndim)
# print(resize_image.shape)



import os

import pickle
from pathlib import Path
from typing import Dict, List, Union

import cv2
import dlib
import numpy as np
from skimage.feature import hog
from tqdm import tqdm

dlib.DLIB_USE_CUDA = False


class PreProcessor:
    def _check_image(self, image: np.ndarray) -> bool:
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Image is not a numpy array: {image}")
        return True

    def gray_image(self, image: np.ndarray) -> np.ndarray:
        self._check_image(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def resize_image(self, image: np.ndarray, size: int) -> np.ndarray:
        self._check_image(image)
        if not isinstance(size, int):
            raise ValueError(f"Size is not an integer: {size}")
        return cv2.resize(image, (size, size))

    def process_image(self, image: np.ndarray) -> np.ndarray:
        image = self.gray_image(image)
        image = self.resize_image(image, 500)
        return image


class FaceDetector:
    def __init__(self):
        self.hog_face_detector = dlib.get_frontal_face_detector()

    def detect_faces(self, image: np.ndarray) -> List[dlib.rectangle]:
        if not isinstance(image, np.ndarray) or image is None:
            raise ValueError(f"Image is not a numpy array: {image}")
        rects = self.hog_face_detector(image, 1)
        return rects


class ImageChopper:
    def chop_image(self, image: np.ndarray, rects: List[dlib.rectangle]) -> Dict[int, np.ndarray]:
        chopped_images = {}
        for i, rect in enumerate(rects):
            crop_img = self._crop_face(image, rect)
            chopped_images[i] = crop_img
        return chopped_images

    def _crop_face(
        self,
        image: np.ndarray,
        rect: dlib.rectangle,
    ) -> np.ndarray:
        if not isinstance(rect, dlib.rectangle) or rect is None:
            raise ValueError(f"Rect is not a dlib rectangle: {rect}")
        if not isinstance(image, np.ndarray) or image is None:
            raise ValueError(f"Image is not a numpy array: {image}")
        if image.ndim != 3:
            raise ValueError(f"Image is not a 3D array: {image}")
        height, width, _ = image.shape
        output_image = image.copy()
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()
        cv2.rectangle(
            output_image,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=(0, 255, 0),
            thickness=width // 200,
        )
        crop_img = output_image[y1:y2, x1:x2]
        return crop_img


class FeatureExtractor:
    def __init__(self):
        self.preprocessor = PreProcessor()
        self.detector = FaceDetector()
        self.image_chopper = ImageChopper()

    def extract_features(self, image: np.ndarray):
        if not isinstance(image, np.ndarray) or image is None:
            raise ValueError(f"Image is not a numpy array: {image}")
        if image.ndim != 3:
            raise ValueError(f"Image is not a 3D array: {image}")
        processed_image = self.preprocessor.process_image(image)
        rects = self.detector.detect_faces(processed_image)
        if rects is None:
            return {}
        chopped_images = self.image_chopper.chop_image(image, rects)
        features = {}
        for i, chopped_image in chopped_images.items():
            crop_img_gray = self.preprocessor.gray_image(chopped_image)
            crop_img_resized = self.preprocessor.resize_image(crop_img_gray, 64)

            feature = hog(
                crop_img_resized,
                orientations=7,
                pixels_per_cell=(8, 8),
                cells_per_block=(4, 4),
                block_norm="L2-Hys",
                transform_sqrt=False,
            )
            feature_reshaped = self._check_ndim(feature)
            features[i] = feature_reshaped
        return features

    def _check_ndim(self, feature: np.ndarray) -> np.ndarray:
        if feature.ndim > 2:
            raise ValueError(f"Feature is not a 2D array: {feature}")
        elif feature.ndim == 1:
            return np.array(feature).reshape(1, -1)
        elif feature.ndim == 2:
            return feature
        raise ValueError(f"Feature is not a 1D or 2D array: {feature}")


class BatchFeatureExtractor:
    def __init__(
        self,
    ):
        self.extractor = FeatureExtractor()

    def process_batch(self, images: Union[np.ndarray, List[np.ndarray]]) -> List[Dict[int, np.ndarray]]:
        features = []
        if isinstance(images, np.ndarray):
            images = [images]
        for image in tqdm(images, total=len(images), desc="detecting faces"):
            feature = self.extractor.extract_features(image)
            if not isinstance(feature, dict):
                raise ValueError(f"Feature is not a dictionaries: {feature} for image: {image}")
            if feature is None:
                raise ValueError(f"Feature is not a dictionary: {feature} for image: {image}")
            features.append(feature)
        return features


class EmotionPredictor:
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> pickle.load:
        if not isinstance(self.model_path, (str, Path)):
            raise ValueError(f"Model path is not a string or path: {model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        print(f"Loading model from {self.model_path}")
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded successfully")
        return model

    def reload_model(self, new_model_path: Union[str, Path] = None):
        """Reload the model, optionally with a new path"""
        if new_model_path is not None:
            self.model_path = new_model_path
        self._model = None  # Force reload on next access
        return self.model

    def predict(self, features: List[Dict[int, np.ndarray]]) -> List[Dict[int, str]]:
        if not isinstance(features, list):
            raise ValueError(f"Features is not a list: {features}")
        if not all(isinstance(feature, dict) for feature in features):
            raise ValueError(f"Not all feature in Features item is a dictionary: {features}")

        predictions = []
        for image_idx, feature_dict in enumerate(features):
            image_predictions = {}
            for face_idx, feature in feature_dict.items():
                try:
                    prediction = self.model.predict(feature)
                    image_predictions[face_idx] = str(prediction[0])
                except Exception as e:
                    raise RuntimeError(f"Prediction failed for image {image_idx}, face {face_idx}: {e}")
            predictions.append(image_predictions)
        return predictions

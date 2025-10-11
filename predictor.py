import os
import sys

sys.path.append(os.getcwd())
from pathlib import Path
from typing import Dict, List, Union
import pickle
from tqdm import tqdm
import cv2
import dlib
import numpy as np
from skimage.feature import hog

dlib.DLIB_USE_CUDA = False


class PreProcessor:

    def __init__(self, image_path: Union[str, Path]):
        self.image_path = self._check_image_path(image_path)
        self.image = self._read_image()

    def _check_image_path(self, image_path: Union[str, Path]) -> None:
        if isinstance(image_path, Path):
            image_path = str(image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        return image_path

    def _check_image(self, image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Image is not a numpy array: {image}")

    def _read_image(self) -> np.ndarray:
        return cv2.imread(self.image_path)

    def gray_image(self, image: np.ndarray) -> np.ndarray:
        self._check_image(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def resize_image(self, image: np.ndarray, size: int) -> np.ndarray:
        self._check_image(image)
        if not isinstance(size, int):
            raise ValueError(f"Size is not an integer: {size}")
        return cv2.resize(image, (size, size))

    def process_image(self) -> np.ndarray:
        image = self.gray_image(self.image)
        image = self.resize_image(image, 500)
        return image


class SingleFaceDetector(PreProcessor):

    def __init__(self, image_path: Union[str, Path]):
        super().__init__(image_path)
        self.hog_face_detector = dlib.get_frontal_face_detector()
        self.processed_image = self.process_image()
        self.rects = self.hog_face_detector(self.processed_image, 1)

    def feature_extraction(self):
        features = {}
        for (i, rect) in enumerate(self.rects):
            crop_img = self._draw_rect(rect)
            crop_img_gray = self.gray_image(crop_img)
            crop_img_resized = self.resize_image(crop_img_gray, 64)

            feature = hog(crop_img_resized,
                          orientations=7,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(4, 4),
                          block_norm='L2-Hys',
                          transform_sqrt=False)
            feature_reshaped = self._check_ndim(feature)
            features[i] = feature_reshaped
        return features

    def _draw_rect(self, rect: dlib.rectangle):
        if not isinstance(rect, dlib.rectangle):
            raise ValueError(f"Rect is not a dlib rectangle: {rect}")
        height, width, _ = self.image.shape
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()
        cv2.rectangle(self.image,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      color=(0, 255, 0),
                      thickness=width // 200)
        crop_img = self.image[y1:y2, x1:x2]
        crop_img = self.resize_image(crop_img, 500)
        return crop_img

    def _check_ndim(self, feature: np.ndarray) -> np.ndarray:
        if feature.ndim > 2:
            raise ValueError(f"Feature is not a 2D array: {feature}")
        elif feature.ndim == 1:
            return np.array(feature).reshape(1, -1)
        elif feature.ndim == 2:
            return feature
        else:
            raise ValueError(f"Feature is not a 1D or 2D array: {feature}")


class FaceDetector():

    def __init__(self, image_paths: Union[str, Path, List[Union[str, Path]]]):
        self.image_paths = image_paths if isinstance(image_paths,
                                                     list) else [image_paths]
        self.single_detector = SingleFaceDetector

    def feature_extraction(self) -> Dict[str, Dict[int, np.ndarray]]:
        features = {}
        for image_path in tqdm(self.image_paths,
                               total=len(self.image_paths),
                               desc="detecting faces"):
            feature = self.single_detector(image_path).feature_extraction()
            features[image_path] = feature
        return features


class EmotionPredictor():

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = model_path
        self.model = self._load_model()

    def _check_model_path(self) -> None:
        if not isinstance(self.model_path, (str, Path)):
            raise ValueError(
                f"Model path is not a string or path: {model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        return True

    def _load_model(self) -> pickle.load:
        if self._check_model_path():
            print(f"Loading model from {self.model_path}")
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully")
            return model
        else:
            raise FileNotFoundError(
                f"Model file does not exist: {self.model_path}")

    def predict(
        self, features: Dict[str,
                             Dict[int,
                                  np.ndarray]]) -> Dict[str, Dict[int, str]]:
        if not isinstance(features, dict):
            raise ValueError(f"Features is not a dictionary: {features}")
        if not isinstance(list(features.items())[0][1], dict):
            raise ValueError(f"Features item is not a dictionary: {features}")

        predictions = {}
        for image_path, feature_dict in features.items():
            for i, feature in feature_dict.items():
                prediction = self.model.predict(feature)
                predictions[image_path] = {i: str(prediction[0])}
        return predictions

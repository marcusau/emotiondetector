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
    """
    A utility class for preprocessing images before feature extraction.
    
    This class provides methods to convert images to grayscale, resize them,
    and perform basic validation checks. It's designed to prepare images
    for face detection and emotion recognition tasks.
    """
    
    def _check_image(self, image: np.ndarray) -> bool:
        """
        Validate that the input is a numpy array.
        
        Args:
            image (np.ndarray): The image to validate.
            
        Returns:
            bool: True if the image is valid.
            
        Raises:
            ValueError: If the image is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Image is not a numpy array: {image}")
        return True

    def gray_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert a BGR image to grayscale.
        
        Args:
            image (np.ndarray): The input BGR image.
            
        Returns:
            np.ndarray: The grayscale version of the image.
        """
        self._check_image(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def resize_image(self, image: np.ndarray, size: int) -> np.ndarray:
        """
        Resize an image to a square with the specified size.
        
        Args:
            image (np.ndarray): The input image to resize.
            size (int): The target size for both width and height.
            
        Returns:
            np.ndarray: The resized square image.
            
        Raises:
            ValueError: If size is not an integer.
        """
        self._check_image(image)
        if not isinstance(size, int):
            raise ValueError(f"Size is not an integer: {size}")
        return cv2.resize(image, (size, size))

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply complete preprocessing pipeline to an image.
        
        This method converts the image to grayscale and resizes it to 500x500 pixels,
        which is the standard preprocessing for face detection in this system.
        
        Args:
            image (np.ndarray): The input image to process.
            
        Returns:
            np.ndarray: The processed grayscale image resized to 500x500.
        """
        image = self.gray_image(image)
        image = self.resize_image(image, 500)
        return image


class FaceDetector:
    """
    A class for detecting faces in images using dlib's HOG-based face detector.
    
    This class provides face detection capabilities using dlib's frontal face detector,
    which is based on Histogram of Oriented Gradients (HOG) features. It's optimized
    for detecting frontal faces in images.
    """
    
    def __init__(self):
        """
        Initialize the face detector with dlib's frontal face detector.
        """
        self.hog_face_detector = dlib.get_frontal_face_detector()

    def detect_faces(self, image: np.ndarray) -> List[dlib.rectangle]:
        """
        Detect faces in the given image.
        
        Args:
            image (np.ndarray): The input image in which to detect faces.
            
        Returns:
            List[dlib.rectangle]: A list of rectangles representing detected faces.
            
        Raises:
            ValueError: If the image is not a numpy array or is None.
        """
        if not isinstance(image, np.ndarray) or image is None:
            raise ValueError(f"Image is not a numpy array: {image}")
        rects = self.hog_face_detector(image, 1)
        return rects


##
class ImageChopper:
    """
    A class for extracting face regions from images.
    
    This class provides functionality to crop detected face regions from images
    based on bounding rectangles. It also adds visual rectangles to the output
    for debugging and visualization purposes.
    """

    def chop_image(self, image: np.ndarray, rects: List[dlib.rectangle]) -> Dict[int, np.ndarray]:
        """
        Extract face regions from an image based on detected face rectangles.
        
        Args:
            image (np.ndarray): The input image containing faces.
            rects (List[dlib.rectangle]): List of rectangles representing detected faces.
            
        Returns:
            Dict[int, np.ndarray]: A dictionary mapping face indices to cropped face images.
        """
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
        """
        Crop a face region from an image and add a visual rectangle.
        
        Args:
            image (np.ndarray): The input image containing the face.
            rect (dlib.rectangle): The rectangle defining the face region.
            
        Returns:
            np.ndarray: The cropped face image with a green rectangle overlay.
            
        Raises:
            ValueError: If rect is not a dlib rectangle, image is not a numpy array,
                       or image is not a 3D array.
        """
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
    """
    A comprehensive feature extraction class for emotion detection.
    
    This class combines preprocessing, face detection, and feature extraction
    to extract HOG (Histogram of Oriented Gradients) features from detected
    faces in images. It's the main component for preparing data for emotion
    classification.
    """
    
    def __init__(self):
        self.preprocessor = PreProcessor()
        self.detector = FaceDetector()
        self.image_chopper = ImageChopper()

    def extract_features(self, image: np.ndarray):
        """
        Extract HOG features from all detected faces in an image.
        
        This method processes an image through the complete pipeline:
        1. Preprocesses the image (grayscale, resize)
        2. Detects faces using HOG detector
        3. Crops face regions
        4. Extracts HOG features from each face
        
        Args:
            image (np.ndarray): The input image to extract features from.
            
        Returns:
            Dict[int, np.ndarray]: A dictionary mapping face indices to HOG features.
                                 Returns empty dict if no faces are detected.
            
        Raises:
            ValueError: If image is not a numpy array, is None, or is not a 3D array.
        """
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
        """
        Ensure feature array has the correct dimensionality for machine learning.
        
        Args:
            feature (np.ndarray): The feature array to check and reshape.
            
        Returns:
            np.ndarray: A 2D feature array suitable for ML models.
            
        Raises:
            ValueError: If feature has more than 2 dimensions or is not 1D/2D.
        """
        if feature.ndim > 2:
            raise ValueError(f"Feature is not a 2D array: {feature}")
        elif feature.ndim == 1:
            return np.array(feature).reshape(1, -1)
        elif feature.ndim == 2:
            return feature
        raise ValueError(f"Feature is not a 1D or 2D array: {feature}")


class BatchFeatureExtractor:
    """
    A batch processing wrapper for feature extraction.
    
    This class provides batch processing capabilities for feature extraction,
    allowing multiple images to be processed efficiently with progress tracking.
    It's designed to handle both single images and batches of images.
    """
    
    def __init__(
        self,
    ):
        self.extractor = FeatureExtractor()

    def process_batch(self, images: Union[np.ndarray, List[np.ndarray]]) -> List[Dict[int, np.ndarray]]:
        """
        Process a batch of images to extract features from all detected faces.
        
        Args:
            images (Union[np.ndarray, List[np.ndarray]]): Single image or list of images to process.
            
        Returns:
            List[Dict[int, np.ndarray]]: List of feature dictionaries, one per input image.
                                      Each dictionary maps face indices to their HOG features.
            
        Raises:
            ValueError: If any extracted feature is not a dictionary or is None.
        """
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
    """
    A class for predicting emotions from extracted features.
    
    This class handles loading and using a trained machine learning model
    to predict emotions from HOG features extracted from face images.
    It supports lazy loading of models and provides prediction capabilities.
    """
    
    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize the emotion predictor with a model path.
        
        Args:
            model_path (Union[str, Path]): Path to the trained model file.
        """
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        """
        Get the loaded model, loading it lazily if necessary.
        
        Returns:
            The loaded machine learning model.
        """
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> pickle.load:
        """
        Load the machine learning model from the specified path.
        
        Returns:
            The loaded pickle model.
            
        Raises:
            ValueError: If model_path is not a string or Path object.
            FileNotFoundError: If the model file does not exist.
        """
        if not isinstance(self.model_path, (str, Path)):
            raise ValueError(f"Model path is not a string or path: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")
        print(f"Loading model from {self.model_path}")
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded successfully")
        return model

    def reload_model(self, new_model_path: Union[str, Path] = None):
        """
        Reload the model, optionally with a new path.
        
        Args:
            new_model_path (Union[str, Path], optional): New path to the model file.
                                                         If None, reloads from current path.
        
        Returns:
            The newly loaded model.
        """
        if new_model_path is not None:
            self.model_path = new_model_path
        self._model = None  # Force reload on next access
        return self.model

    def predict(self, features: List[Dict[int, np.ndarray]]) -> List[Dict[int, str]]:
        """
        Predict emotions from extracted features.
        
        Args:
            features (List[Dict[int, np.ndarray]]): List of feature dictionaries,
                                                  one per image, mapping face indices to HOG features.
        
        Returns:
            List[Dict[int, str]]: List of prediction dictionaries, one per image,
                                mapping face indices to predicted emotion strings.
        
        Raises:
            ValueError: If features is not a list or contains non-dictionary items.
            RuntimeError: If prediction fails for any face.
        """
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

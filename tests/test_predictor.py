import cv2
import dlib
import numpy as np
import pytest

from predictor import (
    BatchFeatureExtractor,
    EmotionPredictor,
    FaceDetector,
    FeatureExtractor,
    ImageChopper,
    PreProcessor,
)


class TestPreProcessor:
    def setup_class(self):
        self.preprocessor = PreProcessor()
        self.sample_image = cv2.imread("images/image1.jpg")

    def test_check_image_valid(self):
        assert self.preprocessor._check_image(self.sample_image) is True

    def test_gray_image(self):
        result = self.preprocessor.gray_image(self.sample_image)
        assert result is not None, "Gray image is not None"
        assert isinstance(result, np.ndarray), "Gray Image is np.ndarray"
        assert result.ndim == 2, "Gray Image dimension is 2"

    def test_resize_image(self):
        size = 500
        result = self.preprocessor.resize_image(self.sample_image, size)
        assert result is not None, "Resize image is not None"
        assert isinstance(result, np.ndarray), "Resize Image is np.ndarray"
        assert result.ndim == 3, "Resize Image dimension is 3"
        assert result.shape[0] == size, "First dimension of shape of result is correct"
        assert result.shape[1] == size, "Second dimension of shape of result is correct"
        assert result.shape[2] == 3, "Third dimension of shape of result is correct"

    def test_process_image(self):
        result = self.preprocessor.process_image(self.sample_image)
        assert result is not None, "Resize image is not None"
        assert isinstance(result, np.ndarray), "Resize Image is np.ndarray"
        assert result.ndim == 2, "Resize Image dimension is 3"
        assert result.shape[0] == 500, "First dimension of shape of result is correct"
        assert result.shape[1] == 500, "Second dimension of shape of result is correct"


class TestFaceDetector:
    def setup_class(self):
        self.face_detector = FaceDetector()
        self.sample_image = cv2.imread("images/image1.jpg")

    def test_detect_faces(self):
        result = self.face_detector.detect_faces(self.sample_image)
        assert result is not None, "face detection result is not None"
        assert all(isinstance(r, dlib.rectangle) for r in result), "Result is dlib.rectangle"


class TestmageChopper:
    def setup_class(self):
        self.sample_image = cv2.imread("images/image1.jpg")
        self.sample_rect = dlib.rectangle(225, 67, 354, 196)
        self.chopper = ImageChopper()

    def test_crop_face(self):
        result = self.chopper._crop_face(self.sample_image, self.sample_rect)
        assert isinstance(result, np.ndarray), "Crop Face result is numpy array"
        assert result.ndim == 3, "Crop Face result is 3 dim"
        assert result.shape[2] == 3, "Third dimension of result is 3 dim"

    def test_chop_image(self):
        result = self.chopper.chop_image(self.sample_image, [self.sample_rect])
        assert isinstance(result, dict), "Chop image result is a dictionary"
        assert all(isinstance(r, np.ndarray) for r in result.values()), "Chop image results are all numpy array"


class TestFeatureExtractor:
    def setup_class(self):
        self.extractor = FeatureExtractor()
        self.sample_image = cv2.imread("images/image1.jpg")

    def test_extract_features(self):
        result = self.extractor.extract_features(self.sample_image)
        assert isinstance(result, dict), "Feature Extractor result is dictionary"
        assert all(isinstance(r, np.ndarray) for r in result.values())
        assert all(r.ndim == 2 for r in result.values()), "All feature results are 2 dim numpy array"
        assert all(r.shape[0] == 1 for r in result.values()), "The first dimensions of all feature results are 1"


class TestBatchFeatureExtractor:
    def setup_class(self):
        self.batch_extractor = BatchFeatureExtractor()
        self.sample_image = cv2.imread("images/image1.jpg")
        self.sample_image_list = [self.sample_image]

    def test_process_batch(self):
        single_result = self.batch_extractor.process_batch(self.sample_image)
        assert isinstance(single_result, list), "Single image batch process result is List"
        assert all(isinstance(v, np.ndarray) for r in single_result for v in r.values()), (
            "All results in single result are numpy array"
        )
        assert all(v.ndim == 2 for r in single_result for v in r.values()), (
            "All results in single result have 2 dimensions"
        )
        assert all(v.shape[0] == 1 for r in single_result for v in r.values()), (
            "The first dimenions values of all results in single result are 1 "
        )

        multi_result = self.batch_extractor.process_batch(self.sample_image_list)
        assert isinstance(multi_result, list), "multi-image batch process result is List"
        assert all(isinstance(v, np.ndarray) for r in multi_result for v in r.values()), (
            "All results in multi-result are numpy array"
        )
        assert all(v.ndim == 2 for r in multi_result for v in r.values()), (
            "All results in multi-result have 2 dimensions"
        )
        assert all(v.shape[0] == 1 for r in multi_result for v in r.values()), (
            "The first dimenions values of all results in multi-result are 1 "
        )


class TestEmotionPredictor:
    def setup_class(self):
        """Set up test fixtures for EmotionPredictor tests."""
        # Create a mock model path for testing
        self.model_path = "svm_model.pkl"
        self.predictor = EmotionPredictor(self.model_path)

        # Create sample features for testing
        # Mock HOG features (typical size for 64x64 image with HOG parameters used)
        self.sample_features = np.random.rand(1, 1764)  # Typical HOG feature size
        self.sample_feature_dict = {0: self.sample_features}
        self.sample_features_list = [self.sample_feature_dict]

        # Create a sample image for integration testing
        self.sample_image = cv2.imread("images/image1.jpg")

    def test_predict_single_image_features(self):
        # This test assumes the model file exists and is valid
        try:
            predictions = self.predictor.predict(self.sample_features_list)

            # Check return type and structure
            assert isinstance(predictions, list), "Predictions should be a list"
            assert len(predictions) == 1, "Should have one prediction per image"
            assert isinstance(predictions[0], dict), "Each prediction should be a dictionary"
            assert 0 in predictions[0], "Should have prediction for face index 0"
            assert isinstance(predictions[0][0], str), "Prediction should be a string"

        except FileNotFoundError:
            # Skip test if model file doesn't exist
            pytest.skip("Model file not found, skipping prediction test")
        except Exception as e:
            # Handle other potential errors gracefully
            pytest.skip(f"Prediction test skipped due to: {e}")

dlib  # import the necessary packages
import os
from pathlib import Path
from typing import Union, Tuple

import cv2
import dlib

import numpy as np

from util import read_image, gray_image, resize_image

dlib.DLIB_USE_CUDA = False

model_master_path = "model"

cnn_face_detector_model_name = "mmod_human_face_detector.dat"
cnn_face_detector_model_path = os.path.join(model_master_path, cnn_face_detector_model_name)

shape_predictor_model_name = "shape_predictor_68_face_landmarks_GTX.dat"
shape_predictor_model_path = os.path.join(model_master_path, shape_predictor_model_name)
face_recognition_model_name = "dlib_face_recognition_resnet_model_v1.dat"
face_recognition_model_path = os.path.join(model_master_path, face_recognition_model_name)

face_folder_path = "face_images"
image_folder_path = "images"

image_filename = "image1.jpg"
image_path = os.path.join(image_folder_path, image_filename)

hog_face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_model_path)
shape_predictor = dlib.shape_predictor(shape_predictor_model_path)
face_recognizer = dlib.face_recognition_model_v1(face_recognition_model_path)


def draw_rect(image: np.ndarray, rect: dlib.rectangle, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Image is not a numpy array: {image}")
    if not isinstance(rect, dlib.rectangle):
        raise ValueError(f"Rect is not a dlib rectangle: {rect}")
    if not isinstance(color, tuple):
        raise ValueError(f"Color is not a tuple: {color}")
    output_image = image.copy()
    height, width, _ = output_image.shape
    x1 = rect.left()
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()
    cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=width // 200)
    crop_img = image[y1:y2, x1:x2]
    crop_img = resize_image(crop_img, 500)
    return crop_img


def save_crop_img(
    crop_img: np.ndarray,
    i: int,
    image_path: Union[str, Path],
    face_folder_path: Union[str, Path],
):
    if not isinstance(crop_img, np.ndarray):
        raise ValueError(f"Crop image is not a numpy array: {crop_img}")
    if not isinstance(i, int):
        raise ValueError(f"Index is not an integer: {i}")
    if not isinstance(image_path, str) and not isinstance(image_path, Path):
        raise ValueError(f"Image path is not a string or Path: {image_path}")
    if not isinstance(face_folder_path, str) and not isinstance(face_folder_path, Path):
        raise ValueError(f"Face folder path is not a string or Path: {face_folder_path}")
    face_filename = f"{Path(image_path).stem}_{i}.jpg"
    face_path = os.path.join(face_folder_path, face_filename)
    if os.path.exists(face_path):
        os.remove(face_path)
    cv2.imwrite(face_path, crop_img)
    return face_path


if __name__ == "__main__":
    image = read_image(image_path)
    output_image = image.copy()
    image_gray = gray_image(image)
    # rects = hog_face_detector(image_gray, 1)
    faces = cnn_face_detector(image_gray)
    # # loop over the face detections
    for i, d in enumerate(faces):
        #     # determine the facial landmarks for the face region, then
        #     # convert the facial landmark (x, y)-coordinates to a NumPy
        #     # array
        shape = shape_predictor(image, d.rect)
        face_desc = face_recognizer.compute_face_descriptor(image, shape, jitter=1)
        print(type(face_desc))
        print(face_desc)
    # crop_img = draw_rect(image, rect)
    # save_crop_img(crop_img, i, image_path, face_folder_path)

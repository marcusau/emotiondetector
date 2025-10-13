import os
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import dlib


# preprocessing: load the input image, resize it, and convert it to grayscale
def read_image(image_path: Union[str, Path]) -> np.ndarray:
    if isinstance(image_path, Path):
        image_path = str(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    image = cv2.imread(image_path)
    return image


def gray_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Image is not a numpy array: {image}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Image is not a numpy array: {image}")
    if not isinstance(size, int):
        raise ValueError(f"Size is not an integer: {size}")
    return cv2.resize(image, (size, size))


def draw_rect(
    image: np.ndarray, rect: dlib.rectangle, color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
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
    cv2.rectangle(
        image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=width // 200
    )
    crop_img = image[y1:y2, x1:x2]
    crop_img = resize_image(crop_img, 500)
    return crop_img


def check_ndim(feature: np.ndarray) -> int:
    if feature.ndim > 2:
        raise ValueError(f"Feature is not a 2D array: {feature}")
    elif feature.ndim == 1:
        return np.array(feature).reshape(1, -1)
    elif feature.ndim == 2:
        return feature
    else:
        raise ValueError(f"Feature is not a 1D or 2D array: {feature}")

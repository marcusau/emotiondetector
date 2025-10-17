import numpy as np
import pytest

from util import gray_image, read_image, resize_image


def test_read_image():
    image_path = "images/image1.jpg"
    image = read_image(image_path)
    assert image is not None, "Image should be loaded successfully."
    assert isinstance(image, np.ndarray), "Loaded image should be a NumPy array."
    assert image.ndim == 3, "Loaded image should be a 3D array."
    assert image.shape[0] > 0, "Loaded image should have a height."
    assert image.shape[1] > 0, "Loaded image should have a width."


def test_gray_image():
    image_path = "images/image1.jpg"
    image = read_image(image_path)
    gray_img = gray_image(image)
    assert gray_img is not None, "Gray image should be loaded successfully."
    assert isinstance(gray_img, np.ndarray), "Gray image should be a NumPy array."
    assert gray_img.ndim == 2, "Gray image should be a 2D array."
    assert gray_img.shape[0] == image.shape[0], "Gray image should have the same height as the original image."
    assert gray_img.shape[1] == image.shape[1], "Gray image should have the same width as the original image."


def test_resize_image():
    image_path = "images/image1.jpg"
    image = read_image(image_path)
    resized_image = resize_image(image, 500)
    assert resized_image is not None, "Resized image should be loaded successfully."
    assert isinstance(resized_image, np.ndarray), "Resized image should be a NumPy array."
    assert resized_image.ndim == 3, "Resized image should be a 3D array."
    assert resized_image.shape[0] == 500, "Resized image should have a height of 100."
    assert resized_image.shape[1] == 500, "Resized image should have a width of 100."
    assert resized_image.shape[2] == 3, "Resized image should have 3 channels."

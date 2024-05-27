import argparse

import cv2
import numpy as np
from numpy.typing import NDArray


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="./models/sam_vit_b_01ec64.pth"
    )
    parser.add_argument("-i", "--input", type=str, default="./data/dog.jpg")
    return parser.parse_args()


def draw_mask(mask: NDArray[np.bool_], image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    color_mask = np.random.randint(0, 256, 3)
    image[mask] = image[mask] * 0.5 + color_mask * (1 - 0.5)
    return image


def draw_box_via_mask(mask, base_image):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    x, y, w, h = cv2.boundingRect(contours[0])
    left_top, right_bottom = (x, y), (x + w, y + h)
    masked_image = base_image.copy()
    cv2.rectangle(masked_image, left_top, right_bottom, (0, 255, 0), 3)
    return masked_image, left_top, right_bottom

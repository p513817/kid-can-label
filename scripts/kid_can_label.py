import argparse
import time

import cv2
import numpy as np
from interative_window import InteractiveWindow
from segment_anything import SamPredictor, sam_model_registry
from utils import draw_box_via_mask, draw_mask


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="./models/sam_vit_b_01ec64.pth"
    )
    parser.add_argument("-i", "--input", type=str, default="./data/dog.jpg")
    return parser.parse_args()


def get_SamPredictor(model_path: str = "/workspace/models/sam_vit_b_01ec64.pth"):
    model_type = "_".join(model_path.split("_")[1:3])
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    return SamPredictor(sam)


class KidCanLabel(InteractiveWindow):
    def __init__(self, model_path: str, input_path: str) -> None:
        super().__init__()
        self.image = cv2.imread(input_path)
        self.draw_image = self.image.copy()
        self.set_background_image(self.image)
        self.predictor = get_SamPredictor(model_path=model_path)
        ts = time.time()
        self.predictor.set_image(self.draw_image)
        print(f"Get Feature Cost: {round(time.time()-ts, 3)}")
        self.add_press_event(ord("s"), self.predict_mask)
        self.add_press_event(ord("b"), self.predict_box)
        self.add_press_event(ord("d"), self.delete_prev_boxes)

    def delete_prev_boxes(self):
        if "boxes" not in self.points:
            self.points["boxes"] = list()
        if self.points["boxes"]:
            self.points["boxes"].pop()
        self.reset_image()
        self.draw_prev_boxes()

    def draw_prev_boxes(self):
        if "boxes" not in self.points:
            self.points["boxes"] = list()
        for left_top, right_bottom in self.points["boxes"]:
            cv2.rectangle(self.draw_image, left_top, right_bottom, (0, 255, 0), 3)

    def predict_mask(self):
        input_point = []
        input_label = []

        for fore_point in self.points[self.FORE]:
            input_point.append(fore_point)
            input_label.append(1)

        for back_point in self.points[self.BACK]:
            input_point.append(back_point)
            input_label.append(0)

        if not (input_point and input_label):
            return
        input_point = np.array(input_point)
        input_label = np.array(input_label)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        mask_input = logits[np.argmax(scores), :, :]
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=True,
        )
        self.reset_detect_point()
        self.draw_image = draw_mask(masks[2], self.draw_image)

    def predict_box(self):
        input_point = []
        input_label = []

        for fore_point in self.points[self.FORE]:
            input_point.append(fore_point)
            input_label.append(1)

        for back_point in self.points[self.BACK]:
            input_point.append(back_point)
            input_label.append(0)
        if not (input_point and input_label):
            return
        input_point = np.array(input_point)
        input_label = np.array(input_label)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        mask_input = logits[np.argmax(scores), :, :]
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=True,
        )
        self.reset_detect_point()
        self.draw_prev_boxes()
        self.draw_image, left_top, right_bottom = draw_box_via_mask(
            masks[2], self.draw_image
        )
        if "boxes" not in self.points:
            self.points["boxes"] = list()
        self.points["boxes"].append((left_top, right_bottom))


if __name__ == "__main__":
    """python3 scripts/kid_can_label.py -m models/sam_vit_b_01ec64.pth -i ./data/dog.jpg"""
    args = build_args()
    kcl = KidCanLabel(model_path=args.model, input_path=args.input)
    kcl.start()

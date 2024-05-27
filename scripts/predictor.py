import sys
import time

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from utils import build_args, draw_box_via_mask, draw_mask


def show_points(coords, labels, marker_size=375):
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    img[pos_points[:, 1], pos_points[:, 0]] = (0, 255, 0)
    img[neg_points[:, 1], neg_points[:, 0]] = (0, 0, 255)
    cv2.imshow("Points Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_box(box):
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
    cv2.imshow("Box Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(model_path: str, image_path: str, device: str = "cuda"):
    model_type = "_".join(model_path.split("_")[1:3])
    image = cv2.imread(image_path)
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)

    ts = time.time()
    predictor = SamPredictor(sam)
    te = time.time()
    print(f"Initialized Predictor cost {te-ts:.3f}s")

    ts = time.time()
    predictor.set_image(image)
    te = time.time()
    print(f"Set image cost {te-ts:.3f}s")

    ts = time.time()
    input_point = np.array([[603, 228]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    te = time.time()
    print(f"Predict cost {te-ts:.3f}s")

    for i, (mask, score) in enumerate(zip(masks, scores)):
        masked_image, left_top, right_bottom = draw_box_via_mask(
            mask=mask, base_image=image
        )
        print(score)
        cv2.imshow("Mask Image", masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = build_args()
    main(args.model, args.input)
    pass

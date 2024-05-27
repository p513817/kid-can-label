import time

import cv2
import numpy as np
from segment_anything import (  # type: ignore
    SamAutomaticMaskGenerator,
    sam_model_registry,
)
from utils import build_args, draw_box_via_mask, draw_mask


def preprocess_for_annotations(annotations):
    """
    sort_annotations _summary_

    Args:
        annotations (_type_): _description_

    Returns:
        _type_: _description_

    Additional:
        Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:
        segmentation : the mask
        area : the area of the mask in pixels
        bbox : the boundary box of the mask in XYWH format
        predicted_iou : the model's own prediction for the quality of the mask
        point_coords : the sampled input point that generated this mask
        stability_score : an additional measure of mask quality
        crop_box : the crop of the image used to generate this mask in XYWH format
    """
    return sorted(annotations, key=(lambda x: x["area"]), reverse=True)


def get_bbox_from_annotations(annotations) -> list:
    """
    get_bbox_from_annotations

    Args:
        annotations (_type_): annotations from SAM

    Returns:
        list: [x1, y1, x2, y2]
    """
    ret = []
    if len(annotations) == 0:
        return ret
    anns = preprocess_for_annotations(annotations=annotations)
    for ann in anns:
        bbox = ann["bbox"]
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        ret.append([x1, y1, x2, y2])
    return ret


def draw_mask_cv(annotations, image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if len(annotations) == 0:
        return image
    anns = preprocess_for_annotations(annotations=annotations)
    for ann in anns:
        m = ann["segmentation"]
        color_mask = np.random.randint(0, 256, 3)
        image[m] = image[m] * alpha + color_mask * (1 - alpha)

    return image


def draw_bbox_cv(annotations, image: np.ndarray) -> np.ndarray:
    if len(annotations) == 0:
        return image
    bbox = get_bbox_from_annotations(annotations=annotations)
    for x1, y1, x2, y2 in bbox:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def main(model_path: str, image_path: str, device: str = "cuda"):
    model_type = "_".join(model_path.split("_")[1:3])
    image = cv2.imread(image_path)

    ts = time.time()
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    te = time.time()
    print(f"Load SAM Cost: {round(te-ts, 3)}s")

    ts = time.time()
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        pred_iou_thresh=0.96,
        stability_score_thresh=0.95,
        points_per_side=10,
        # crop_n_points_downscale_factor=2,
    )

    te = time.time()
    print(f"Init Generator Cost: {te-ts:.3f}s")

    ts = time.time()
    masks = mask_generator.generate(image)
    te = time.time()
    print(f"Gen Mask Cost: {te-ts:.3f}s")

    ts = time.time()
    mask_result = draw_mask_cv(masks, image.copy())
    bbox_result = draw_bbox_cv(masks, image.copy())
    te = time.time()
    print(f"Draw Mask Cost: {te-ts:.3f}s")

    result = cv2.hconcat([mask_result, bbox_result])
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", result)
    while cv2.waitKey(1) != 27:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = build_args()
    main(args.model, args.input)

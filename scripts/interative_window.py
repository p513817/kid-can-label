from collections import defaultdict
from typing import List, Literal, Union

import cv2
import numpy as np


class InteractiveWindow:
    FORE = "foreground"
    BACK = "background"

    def __init__(self) -> None:
        self.win_name: str = "KID CAN LABEL"
        self.points: dict = defaultdict(list)
        self.operations: list = list()
        self.image: np.ndarray = np.ones((500, 500, 3), dtype=np.uint8)
        self.key_event: dict = {
            ord("c"): self.reset,
            ord("z"): self.rollback,
            ord("q"): lambda: "exit",
        }

        self.reset()
        self._init_windows()

    def _init_windows(self):
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.mouse_callback)

    def reset_detect_point(self):
        self.points[self.FORE] = list()
        self.points[self.BACK] = list()

    def reset(self):
        self.points = defaultdict(list)
        self.operations = list()
        self.reset_image()

    def reset_image(self):
        self.draw_image = self.image.copy()

    def add_operations(self, operation):
        assert operation in [self.FORE, self.BACK], "Operation is invalid"
        while len(self.operations) >= 10:
            self.operations.pop(0)
        self.operations.append(operation)

    def add_foreground_point(self, x, y):
        self.points[self.FORE].append((x, y))
        self.add_operations(self.FORE)

    def add_background_point(self, x, y):
        self.points[self.BACK].append((x, y))
        self.add_operations(self.BACK)

    def rollback(self):
        if not self.operations:
            return
        latest = self.operations.pop()
        self.points[latest].pop()

    def mouse_callback(self, event, x, y, flags, param):
        global left_clicks, right_clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_foreground_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.add_background_point(x, y)

    def set_background_image(self, image: Union[np.ndarray, None]):
        if image is None:
            return
        self.image = image

    def add_press_event(self, key, func):
        if isinstance(key, str):
            key = ord(key)
        if key in self.key_event:
            raise KeyError("Key already used")
        self.key_event[key] = func

    def press_event(self, key):
        if key in self.key_event:
            return self.key_event[key]()
        return None

    def start(self):
        while True:
            temp_img = self.draw_image.copy()
            for ground, points in self.points.items():
                if ground not in [self.FORE, self.BACK]:
                    continue
                color = (0, 255, 0) if ground == self.FORE else (0, 0, 255)
                for point in points:
                    cv2.circle(temp_img, point, 5, color, -1)
            cv2.imshow(self.win_name, temp_img)

            key = cv2.waitKey(1) & 0xFF
            if self.press_event(key):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        image = cv2.imread(sys.argv[1])
    else:
        image = None

    kid = InteractiveWindow()
    kid.set_background_image(image)
    kid.start()

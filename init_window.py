import json
import time
from pathlib import Path

import cv2
import mss
import numpy as np


CONFIG_PATH = Path("woa_config.json")


def grab_fullscreen():
    with mss.mss() as sct:
        # use primary monitor
        mon = sct.monitors[1]
        img = np.array(sct.grab(mon))  # BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img, (mon["left"], mon["top"])


def select_region(img):
    clone = img.copy()
    selecting = False
    pt1 = None
    rect = None

    def on_mouse(event, x, y, flags, param):
        nonlocal selecting, pt1, rect, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            pt1 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            img2 = img.copy()
            cv2.rectangle(img2, pt1, (x, y), (0, 255, 0), 2)
            clone = img2
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            pt2 = (x, y)
            x0, y0 = pt1
            x1, y1 = pt2
            x_min, y_min = min(x0, x1), min(y0, y1)
            x_max, y_max = max(x0, x1), max(y0, y1)
            rect = (x_min, y_min, x_max - x_min, y_max - y_min)
            img2 = img.copy()
            cv2.rectangle(img2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            clone = img2

    win = "Drag to select region, ENTER=confirm, R=reset, Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1200, int(1200 * img.shape[0] / max(img.shape[1], 1)))
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, clone)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 or key == 10:  # Enter
            break
        elif key in (ord("r"), ord("R")):
            clone = img.copy()
            rect = None
        elif key in (ord("q"), ord("Q"), 27):  # q or ESC
            rect = None
            break

    cv2.destroyWindow(win)
    return rect


def save_config(abs_rect):
    # abs_rect: (left, top, width, height) in absolute screen coords
    data = {
        "EMU_REGION": {
            "left": abs_rect[0],
            "top": abs_rect[1],
            "width": abs_rect[2],
            "height": abs_rect[3],
        }
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(
        f"Saved region to {CONFIG_PATH.resolve()}:\n{json.dumps(data, indent=2, ensure_ascii=False)}"
    )


def load_config():
    if not CONFIG_PATH.exists():
        print("No existing region config found, launching selector...")
        init()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    region = cfg["EMU_REGION"]
    return (region["left"], region["top"], region["width"], region["height"])


def preview_capture(abs_rect, seconds=3, fps=5):

    with mss.mss() as sct:
        left, top, w, h = abs_rect
        t_end = time.time() + seconds
        cv2.namedWindow("Region Preview", cv2.WINDOW_NORMAL)
        try:
            while time.time() < t_end:
                img = np.array(
                    sct.grab({"left": left, "top": top, "width": w, "height": h})
                )
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                cv2.imshow("Region Preview", img)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
                    break
        finally:
            cv2.destroyWindow("Region Preview")


def init():
    print("Taking fullscreen screenshot...")
    full, (off_x, off_y) = grab_fullscreen()
    print("Please DRAG a rectangle that tightly bounds the game view.")
    rect_rel = select_region(full)  # relative to screenshot
    if rect_rel is None:
        print("No region selected. Exiting.")
        return
    # Convert to absolute screen coords
    left = off_x + rect_rel[0]
    top = off_y + rect_rel[1]
    abs_rect = (left, top, rect_rel[2], rect_rel[3])
    print("Absolute region:", abs_rect)
    save_config(abs_rect)
    print("Done. Now other scripts can read EMU_REGION from woa_config.json.")

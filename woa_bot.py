from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import Counter

import cv2
import numpy as np
from mss import mss
from pynput import keyboard

import init_window
import mouse_action
import scenarios


# Capture area ratios for the right notification column.
RIGHT_PANEL_RELATIVE = (0.72, 0.0, 0.28, 1.0)
ATTENTION_TEMPLATE_PATH = Path("assets/templates/attention_icon.png")
CONFIG_PATH = Path("woa_config.json")

# Template-matching parameters.
MATCH_THRESHOLD = 0.7
MAX_VISIBLE_ICONS = 10

# Scroll behaviour.
SCROLL_STEP_COUNT = 6
SCROLL_DELTA = 180
SCROLL_ATTEMPT_LIMIT = 6
SCROLL_SAMPLE_DELAY = 0.06
SCROLL_SETTLE_PAUSE = 0.2
SCROLL_PAUSE = 0.3

# Loop pacing.
NO_ICON_PAUSE = 0.6
CLICK_PAUSE = 1.0

# Misc helpers.
SKIP_CARD_X_TOLERANCE = 12


def main() -> None:
    """Load user configuration and start the hotkey-controlled loop."""
    region = init_window.load_config()
    _run_with_toggle(region)


def _run_with_toggle(region: Dict[str, int] | Tuple[int, int, int, int]) -> None:
    """Toggle the automation on Ctrl+Space."""
    toggle_state = {
        "active": False,
        "debounce": False,
        "base_icon_x": _load_base_icon_x(),
    }
    pressed_keys: set[object] = set()

    def has_ctrl() -> bool:
        return (
            keyboard.Key.ctrl_l in pressed_keys or keyboard.Key.ctrl_r in pressed_keys
        )

    def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
        pressed_keys.add(key)
        if key == keyboard.Key.space and has_ctrl() and not toggle_state["debounce"]:
            toggle_state["active"] = not toggle_state["active"]
            toggle_state["debounce"] = True
            if not toggle_state["active"]:
                print("Play stopped.")
                print("Press Ctrl + Enter to begin.")

    def on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
        pressed_keys.discard(key)
        if key == keyboard.Key.space:
            toggle_state["debounce"] = False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("Press Ctrl + Enter to begin.")

    try:
        while True:
            if toggle_state["active"]:
                play_game(region, toggle_state)
            time.sleep(0.1)
    finally:
        listener.stop()


def play_game(
    region: Dict[str, int] | Tuple[int, int, int, int],
    toggle_state: Dict[str, bool] | None = None,
) -> None:
    left, top, width, height = _normalize_region(region)
    if width <= 0 or height <= 0:
        raise ValueError("Configured capture region has non-positive dimensions.")
    template = _load_template(ATTENTION_TEMPLATE_PATH)
    panel_bbox = _relative_bbox(left, top, width, height, RIGHT_PANEL_RELATIVE)
    full_bbox = {"left": left, "top": top, "width": width, "height": height}
    with mss() as sct:
        # Continually process the right panel while hotkey remains active.
        while toggle_state is None or toggle_state.get("active", False):
            panel_img = _capture_area(sct, panel_bbox)
            icons = _detect_attention_icons(panel_img, template)
            if not icons:
                # Nothing to handle yet; wait briefly before retry.
                time.sleep(NO_ICON_PAUSE)
                continue
            if len(icons) >= MAX_VISIBLE_ICONS:
                # Too many icons; scroll to reveal the bottom entries.
                top_x, top_y = icons[0]
                mouse_action.move_to(
                    panel_bbox["left"] + top_x, panel_bbox["top"] + top_y, duration=0.12
                )
                time.sleep(SCROLL_PAUSE)
                icons = _scroll_to_bottom(sct, panel_bbox, template)
                if not icons:
                    time.sleep(NO_ICON_PAUSE)
                    continue
                panel_img = _capture_area(sct, panel_bbox)
                icons = _detect_attention_icons(panel_img, template)
                if not icons:
                    time.sleep(NO_ICON_PAUSE)
                    continue

            base_icon_x = None
            if toggle_state is not None:
                base_icon_x = toggle_state.get("base_icon_x")
                if base_icon_x is None and len(icons) > 1:
                    inferred = _infer_base_icon_x(icons)
                    if inferred is not None:
                        toggle_state["base_icon_x"] = inferred
                        _save_base_icon_x(inferred)
                        base_icon_x = inferred

            if len(icons) > 1:
                if _should_skip_card_click(icons, base_icon_x):
                    print("Skipped card click due to offset; evaluating scenarios.")
                    handled = _process_active_scenarios(sct, full_bbox)
                    if handled:
                        time.sleep(1.0)
                    continue
            else:
                if base_icon_x is not None and _should_skip_card_click(
                    icons, base_icon_x
                ):
                    print(
                        "Single attention card already selected; evaluating scenarios."
                    )
                    handled = _process_active_scenarios(sct, full_bbox)
                    if handled:
                        time.sleep(1.0)
                    continue
            target_x, target_y = icons[-1]
            # Click the bottom-most attention card.
            absolute_x = panel_bbox["left"] + target_x
            absolute_y = panel_bbox["top"] + target_y
            mouse_action.click_at(
                absolute_x,
                absolute_y,
                move_duration=0.12,
                label="attention card",
            )
            print("Clicked event. Start handeling...")
            time.sleep(CLICK_PAUSE)
            handled = _process_active_scenarios(sct, full_bbox)
            if handled:
                time.sleep(1.0)


def _normalize_region(
    region: Dict[str, int] | Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    if isinstance(region, dict):
        return (
            int(region["left"]),
            int(region["top"]),
            int(region["width"]),
            int(region["height"]),
        )
    left, top, width, height = region
    return int(left), int(top), int(width), int(height)


def _relative_bbox(
    left: int,
    top: int,
    width: int,
    height: int,
    ratios: Tuple[float, float, float, float],
) -> Dict[str, int]:
    rel_left, rel_top, rel_width, rel_height = ratios
    panel_left = left + int(width * rel_left)
    panel_top = top + int(height * rel_top)
    panel_width = int(width * rel_width)
    panel_height = int(height * rel_height)
    return {
        "left": panel_left,
        "top": panel_top,
        "width": panel_width,
        "height": panel_height,
    }


def _capture_area(sct: mss, bbox: Dict[str, int]) -> np.ndarray:
    grab = sct.grab(bbox)
    frame = np.array(grab)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def _capture_panel(sct: mss, bbox: Dict[str, int]) -> np.ndarray:
    """Backward-compatible alias."""
    return _capture_area(sct, bbox)


def _detect_attention_icons(
    panel_img: np.ndarray, template: np.ndarray
) -> List[Tuple[int, int]]:
    # Match the yellow icon template across the cropped panel.
    match = cv2.matchTemplate(panel_img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(match >= MATCH_THRESHOLD)
    template_h, template_w = template.shape[:2]
    tolerance_x = max(4, template_w // 2)
    tolerance_y = max(4, template_h // 2)
    icons: List[Tuple[int, int]] = []
    for x, y in zip(loc[1], loc[0]):
        center = (int(x + template_w // 2), int(y + template_h // 2))
        if _is_new_center(center, icons, tolerance_x, tolerance_y):
            icons.append(center)
    icons.sort(key=lambda point: point[1])
    return icons


def _is_new_center(
    candidate: Tuple[int, int],
    existing: List[Tuple[int, int]],
    tol_x: int,
    tol_y: int,
) -> bool:
    cx, cy = candidate
    for ex, ey in existing:
        if abs(cx - ex) <= tol_x and abs(cy - ey) <= tol_y:
            return False
    return True


def _should_skip_card_click(
    icons: List[Tuple[int, int]], base_icon_x: int | None = None
) -> bool:
    """Detect when bottom icon is already selected."""
    bottom_x = icons[-1][0]
    if base_icon_x is not None:
        return abs(bottom_x - base_icon_x) > SKIP_CARD_X_TOLERANCE
    # Fall back to comparing against other icons when no baseline is known.
    if len(icons) <= 1:
        return False
    return all(abs(bottom_x - x) > SKIP_CARD_X_TOLERANCE for x, _ in icons[:-1])


def _process_active_scenarios(sct: mss, bbox: Dict[str, int]) -> bool:
    """Capture the full region and run scenario-specific handlers."""
    capture_fn = lambda: _capture_area(sct, bbox)
    try:
        frame = capture_fn()
    except Exception:
        return False

    context = scenarios.ScenarioContext(frame, bbox, capture_fn)
    handlers = (
        scenarios.select_stand,
        scenarios.approve_landing,
        scenarios.assign_ground_service,
        scenarios.ground_service_in_progress,
        scenarios.ground_service_complete,
        scenarios.ground_service_reward,
        scenarios.pushback,
        scenarios.cross_runway,
        scenarios.taxi_to_hold,
        scenarios.takeoff,
    )

    for index, handler in enumerate(handlers):
        if index > 0:
            context.refresh()
        try:
            if handler(context):
                return True
        except FileNotFoundError as exc:
            print(f"[Scenario] Missing template: {exc}")
    return False


def _scroll_to_bottom(
    sct: mss, bbox: Dict[str, int], template: np.ndarray
) -> List[Tuple[int, int]]:
    panel_img = _capture_area(sct, bbox)
    icons = _detect_attention_icons(panel_img, template)
    if not icons:
        return []

    attempts = 0

    while attempts < SCROLL_ATTEMPT_LIMIT:
        mouse_action.scroll_down(
            steps=SCROLL_STEP_COUNT,
            step_delta=SCROLL_DELTA,
            pause=0.12,
        )
        time.sleep(SCROLL_SAMPLE_DELAY)

        panel_img = _capture_area(sct, bbox)
        icons = _detect_attention_icons(panel_img, template)
        if not icons:
            return []

        current_count = len(icons)
        if current_count <= MAX_VISIBLE_ICONS - 2:
            time.sleep(SCROLL_SETTLE_PAUSE)
            panel_img = _capture_area(sct, bbox)
            icons = _detect_attention_icons(panel_img, template)
            return icons

        attempts += 1
        time.sleep(SCROLL_PAUSE)

    return icons


def _infer_base_icon_x(icons: List[Tuple[int, int]]) -> int | None:
    """Infer the common X coordinate for unselected attention icons."""
    if len(icons) <= 1:
        return None
    counts = Counter(x for x, _ in icons)
    if not counts:
        return None
    common_x, count = counts.most_common(1)[0]
    if count >= max(1, len(icons) - 1):
        return common_x
    return None


def _load_base_icon_x() -> int | None:
    data = _load_config_data()
    value = data.get("ATTENTION_BASE_X")
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _save_base_icon_x(value: int) -> None:
    data = _load_config_data()
    if data.get("ATTENTION_BASE_X") == value:
        return
    data["ATTENTION_BASE_X"] = int(value)
    _save_config_data(data)
    print(f"[Info] Saved attention icon base X: {value}")


def _load_config_data() -> Dict[str, object]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return {}


def _save_config_data(data: Dict[str, object]) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_template(path: Path) -> np.ndarray:
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Template image not found at {path}")
    return template


if __name__ == "__main__":
    main()

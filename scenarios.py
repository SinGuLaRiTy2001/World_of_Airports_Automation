from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set, Tuple

import cv2
import numpy as np

import mouse_action

_GREEN = "\033[92m"
_RESET = "\033[0m"

try:  # Optional dependency; OCR fallback will be disabled if missing.
    import pytesseract
    from pytesseract import Output as TesseractOutput
except ImportError:
    print(f"{_GREEN}[Warning]{_RESET} Pytesseract not imported; OCR fallback disabled.")
    pytesseract = None
    TesseractOutput = None


TEMPLATES_DIR = Path("assets/templates")

SELECT_STAND_TITLE = "select_stand_title.png"
SELECT_STAND_AVAILABLE = "stand_available.png"
SELECT_STAND_CONFIRM = "stand_confirm.png"
SELECT_STAND_TITLE_TEXT = "\u9009\u62e9\u673a\u4f4d"
SELECT_STAND_AVAILABLE_TEXT = "\u53ef\u7528"
SELECT_STAND_CONFIRM_TEXT = "\u786e\u8ba4"

LANDING_APPROVE_BUTTON = "allow_landing.png"
LANDING_APPROVE_TEXT = "\u51c6\u8bb8\u7740\u964d"

GROUND_SERVICE_TITLE = "ground_service_title.png"
GROUND_SERVICE_TITLE_TEXT = "\u5730\u52e4\u4eba\u5458"
GROUND_SERVICE_ANCHOR = "ground_service_anchor.png"
GROUND_SERVICE_START = "ground_service_start.png"
GROUND_SERVICE_START_TEXT = "\u5f00\u59cb\u5730\u9762\u670d\u52a1"
GROUND_SERVICE_IN_PROGRESS_TEXT = "\u5904\u7406\u4e2d"
GROUND_SERVICE_IN_PROGRESS = "ground_in_progress.png"
GROUND_SERVICE_TASK_ICONS = {
    "deboard": "ground_deboard.png",
    "unload_baggage": "ground_unload_baggage.png",
    "refuel": "ground_refuel.png",
    "waste": "ground_waste.png",
    "cleaning": "ground_cleaning.png",
    "water": "ground_water.png",
    "catering": "ground_catering.png",
    "load_baggage": "ground_load_baggage.png",
    "boarding": "ground_boarding.png",
    "unload_cargo": "ground_unload_cargo.png",
    "load_cargo": "ground_load_cargo.png",
}
GROUND_SERVICE_COMPLETE = "ground_service_complete.png"
GROUND_SERVICE_COMPLETE_TEXT = "\u5b8c\u6210\u5730\u9762\u670d\u52a1\u4fdd\u969c"
GROUND_SERVICE_REWARD = "ground_service_reward.png"
GROUND_SERVICE_REWARD_TEXT = "\u9886\u53d6\u5956\u52b1"
GROUND_SERVICE_CLAIM_VARIANTS = (
    "ground_service_claim.png",
    "ground_service_claim_alt.png",
)
GROUND_SERVICE_CLAIM_TEXTS = (
    "\u9886\u53d6",
    "\u83b7\u5f97\u5956\u52b1\u5e76\u5347\u7ea7",
)

PUSHBACK_BUTTON = "pushback.png"
PUSHBACK_TEXT = "\u63a8\u51fa"

CROSS_RUNWAY_BUTTON = "cross_runway.png"
CROSS_RUNWAY_TEXT = "\u7a7f\u8d8a\u8dd1\u9053"

TAXI_TO_HOLD_BUTTON = "taxi_to_hold.png"
TAXI_TO_HOLD_TEXT = "\u8fdb\u5165\u8dd1\u9053\u7b49\u5f85"

TAKEOFF_BUTTON = "takeoff.png"
TAKEOFF_TEXT = "\u8d77\u98de"

_TEMPLATE_CACHE: Dict[str, np.ndarray] = {}
_OCR_SCALE = 1.6
_OCR_MIN_CONF = 65.0
ENABLE_OCR = False


class ScenarioContext:
    """Bundle of frame data and helpers reused across scenario handlers."""

    def __init__(
        self,
        frame: np.ndarray,
        region: Dict[str, int],
        capture_fn: Callable[[], np.ndarray],
    ) -> None:
        self.frame = frame
        self.region = region
        self._capture_fn = capture_fn

    def refresh(self, delay: float = 0.0) -> np.ndarray:
        if delay > 0.0:
            time.sleep(delay)
        self.frame = self._capture_fn()
        return self.frame


def select_stand(context: ScenarioContext) -> bool:
    """Handle the stand selection screen."""
    frame = context.frame
    region = context.region
    if frame is None or frame.size == 0:
        return False

    if not _has_select_stand_title(frame):
        return False

    available_hits = _match_template(
        frame,
        SELECT_STAND_AVAILABLE,
        threshold=0.78,
        debug_label="stand_available",
    )
    if not available_hits:
        available_hits = _ocr_find_keywords(
            frame,
            {SELECT_STAND_AVAILABLE_TEXT},
            lang="chi_sim",
            debug_label="stand_available_ocr",
        )
    if not available_hits:
        return False

    leftmost_available = min(available_hits, key=lambda pt: pt[0])
    abs_leftmost = _to_absolute(region, leftmost_available)
    mouse_action.click_at(
        abs_leftmost[0],
        abs_leftmost[1],
        move_duration=0.15,
        label="select-stand available",
    )

    frame = context.refresh(delay=0.2)

    confirm_hits = _match_template(
        frame,
        SELECT_STAND_CONFIRM,
        threshold=0.78,
        debug_label="stand_confirm",
    )
    if not confirm_hits:
        confirm_hits = _ocr_find_keywords(
            frame,
            {SELECT_STAND_CONFIRM_TEXT},
            lang="chi_sim",
            debug_label="stand_confirm_ocr",
        )
    if not confirm_hits:
        return False

    confirm_center = confirm_hits[0]
    abs_confirm = _to_absolute(region, confirm_center)
    mouse_action.click_at(
        abs_confirm[0],
        abs_confirm[1],
        move_duration=0.12,
        label="select-stand confirm",
    )
    time.sleep(0.2)
    return True


def approve_landing(context: ScenarioContext) -> bool:
    """Handle the landing approval screen by clicking the '准许着陆' control."""
    frame = context.frame
    region = context.region

    hits = _match_template(
        frame, LANDING_APPROVE_BUTTON, threshold=0.82, debug_label="allow_landing"
    )
    if not hits:
        hits = _ocr_find_keywords(
            frame,
            {LANDING_APPROVE_TEXT},
            lang="chi_sim",
            debug_label="allow_landing_ocr",
        )
    if not hits:
        return False

    abs_point = _to_absolute(region, hits[0])
    mouse_action.click_at(
        abs_point[0],
        abs_point[1],
        move_duration=0.12,
        label="allow landing",
    )
    time.sleep(0.2)
    return True


def assign_ground_service(context: ScenarioContext) -> bool:
    """Handle the ground crew assignment screen."""
    frame = context.frame
    region = context.region

    title_hits = _match_template(
        frame, GROUND_SERVICE_TITLE, threshold=0.75, debug_label="ground_service_title"
    )
    if not title_hits:
        title_hits = _ocr_find_keywords(
            frame,
            {GROUND_SERVICE_TITLE_TEXT},
            lang="chi_sim",
            debug_label="ground_service_title_ocr",
        )
    if not title_hits:
        return False

    anchor_hits = _match_template(
        frame,
        GROUND_SERVICE_ANCHOR,
        threshold=0.78,
        debug_label="ground_service_anchor",
    )
    if not anchor_hits:
        return False

    anchor = min(anchor_hits, key=lambda pt: pt[0])
    abs_anchor = _to_absolute(region, anchor)
    target_x = region["left"] + region["width"] // 2
    target_y = abs_anchor[1]

    mouse_action.move_to(abs_anchor[0], abs_anchor[1], duration=0.08)
    time.sleep(0.05)
    mouse_action.mouse_down()
    mouse_action.move_to(target_x, target_y, duration=0.18)
    mouse_action.mouse_up()
    time.sleep(0.2)

    frame = context.refresh(delay=0.2)

    start_hits = _match_template(
        frame,
        GROUND_SERVICE_START,
        threshold=0.8,
        debug_label="ground_service_start",
    )
    if not start_hits:
        start_hits = _ocr_find_keywords(
            frame,
            {GROUND_SERVICE_START_TEXT},
            lang="chi_sim",
            debug_label="ground_service_start_ocr",
        )
    if not start_hits:
        return False

    start_point = start_hits[0]
    abs_start = _to_absolute(region, start_point)
    mouse_action.click_at(
        abs_start[0],
        abs_start[1],
        move_duration=0.12,
        label="start ground service",
    )
    time.sleep(0.2)
    return True


def ground_service_in_progress(context: ScenarioContext) -> bool:
    """Handle the ground crew in-progress screen."""
    frame = context.frame
    region = context.region

    in_progress_hits = _match_template(
        frame,
        GROUND_SERVICE_IN_PROGRESS,
        threshold=0.78,
        debug_label="ground_in_progress",
    )
    if not in_progress_hits:
        in_progress_hits = _ocr_find_keywords(
            frame,
            {GROUND_SERVICE_IN_PROGRESS_TEXT},
            lang="chi_sim",
            debug_label="ground_service_in_progress_ocr",
        )
    if not in_progress_hits:
        return False

    activated = False
    for name, template_file in GROUND_SERVICE_TASK_ICONS.items():
        hits = _match_template(
            frame,
            template_file,
            threshold=0.78,
            debug_label=f"ground_service_{name}",
        )
        if not hits:
            continue
        for center in hits:
            abs_pt = _to_absolute(region, center)
            mouse_action.click_at(
                abs_pt[0],
                abs_pt[1],
                move_duration=0.09,
                label=f"ground task-{name}",
            )
            activated = True

    return activated


def ground_service_complete(context: ScenarioContext) -> bool:
    """Handle the ground service completion prompt."""
    frame = context.frame
    region = context.region

    complete_hits = _match_template(
        frame,
        GROUND_SERVICE_COMPLETE,
        threshold=0.78,
        debug_label="ground_service_complete",
    )
    if not complete_hits:
        complete_hits = _ocr_find_keywords(
            frame,
            {GROUND_SERVICE_COMPLETE_TEXT},
            lang="chi_sim",
            debug_label="ground_service_complete_ocr",
        )
    if not complete_hits:
        return False

    abs_complete = _to_absolute(region, complete_hits[0])
    mouse_action.click_at(
        abs_complete[0],
        abs_complete[1],
        move_duration=0.12,
        label="complete ground service",
    )
    return True


def ground_service_reward(context: ScenarioContext) -> bool:
    """Handle optional reward collection after ground services."""
    frame = context.frame
    region = context.region

    reward_hits = _match_template(
        frame,
        GROUND_SERVICE_REWARD,
        threshold=0.78,
        debug_label="ground_service_reward",
    )
    if not reward_hits:
        reward_hits = _ocr_find_keywords(
            frame,
            {GROUND_SERVICE_REWARD_TEXT},
            lang="chi_sim",
            debug_label="ground_service_reward_ocr",
        )
    if not reward_hits:
        return False

    abs_reward = _to_absolute(region, reward_hits[0])
    mouse_action.click_at(
        abs_reward[0],
        abs_reward[1],
        move_duration=0.12,
        label="collect reward",
    )

    frame = context.refresh(delay=0.5)
    claim_hits: List[Tuple[int, int]] = []
    for variant in GROUND_SERVICE_CLAIM_VARIANTS:
        hits = _match_template(
            frame,
            variant,
            threshold=0.78,
            debug_label=f"ground_service_claim:{variant}",
        )
        if hits:
            claim_hits.extend(hits)

    if not claim_hits:
        claim_hits = _ocr_find_keywords(
            frame,
            set(GROUND_SERVICE_CLAIM_TEXTS),
            lang="chi_sim",
            debug_label="ground_service_claim_ocr",
        )
    if not claim_hits:
        return True

    abs_claim = _to_absolute(region, claim_hits[0])
    mouse_action.click_at(
        abs_claim[0],
        abs_claim[1],
        move_duration=0.12,
        label="claim reward",
    )
    time.sleep(2.0)
    return True


def pushback(context: ScenarioContext) -> bool:
    """Handle the pushback confirmation."""
    frame = context.frame
    region = context.region

    push_hits = _match_template(
        frame, PUSHBACK_BUTTON, threshold=0.78, debug_label="pushback"
    )
    if not push_hits:
        push_hits = _ocr_find_keywords(
            frame,
            {PUSHBACK_TEXT},
            lang="chi_sim",
            debug_label="pushback_ocr",
        )
    if not push_hits:
        return False

    abs_push = _to_absolute(region, push_hits[0])
    mouse_action.click_at(
        abs_push[0],
        abs_push[1],
        move_duration=0.12,
        label="pushback",
    )
    time.sleep(0.2)
    return True


def cross_runway(context: ScenarioContext) -> bool:
    """Handle the runway crossing confirmation."""
    frame = context.frame
    region = context.region

    cross_hits = _match_template(
        frame, CROSS_RUNWAY_BUTTON, threshold=0.78, debug_label="cross_runway"
    )
    if not cross_hits:
        cross_hits = _ocr_find_keywords(
            frame,
            {CROSS_RUNWAY_TEXT},
            lang="chi_sim",
            debug_label="cross_runway_ocr",
        )
    if not cross_hits:
        return False

    abs_cross = _to_absolute(region, cross_hits[0])
    mouse_action.click_at(
        abs_cross[0],
        abs_cross[1],
        move_duration=0.12,
        label="cross runway",
    )
    time.sleep(0.2)
    return True


def taxi_to_hold(context: ScenarioContext) -> bool:
    """Handle the 'enter runway and hold' confirmation."""
    frame = context.frame
    region = context.region

    hold_hits = _match_template(
        frame, TAXI_TO_HOLD_BUTTON, threshold=0.78, debug_label="taxi_to_hold"
    )
    if not hold_hits:
        hold_hits = _ocr_find_keywords(
            frame,
            {TAXI_TO_HOLD_TEXT},
            lang="chi_sim",
            debug_label="taxi_to_hold_ocr",
        )
    if not hold_hits:
        return False

    abs_hold = _to_absolute(region, hold_hits[0])
    mouse_action.click_at(
        abs_hold[0],
        abs_hold[1],
        move_duration=0.12,
        label="taxi to hold",
    )
    time.sleep(0.2)
    return True


def takeoff(context: ScenarioContext) -> bool:
    """Handle the takeoff confirmation button."""
    frame = context.frame
    region = context.region

    takeoff_hits = _match_template(
        frame, TAKEOFF_BUTTON, threshold=0.78, debug_label="takeoff"
    )
    if not takeoff_hits:
        takeoff_hits = _ocr_find_keywords(
            frame,
            {TAKEOFF_TEXT},
            lang="chi_sim",
            debug_label="takeoff_ocr",
        )
    if not takeoff_hits:
        return False

    abs_takeoff = _to_absolute(region, takeoff_hits[0])
    mouse_action.click_at(
        abs_takeoff[0],
        abs_takeoff[1],
        move_duration=0.12,
        label="takeoff",
    )
    time.sleep(0.2)
    return True


def _has_select_stand_title(frame: np.ndarray) -> bool:
    title_hits = _match_template(
        frame, SELECT_STAND_TITLE, threshold=0.75, debug_label="select_stand_title"
    )
    if title_hits:
        return True
    title_hits = _ocr_find_keywords(
        frame,
        {SELECT_STAND_TITLE_TEXT},
        lang="chi_sim",
        debug_label="select_stand_title_ocr",
    )
    return bool(title_hits)


def _match_template(
    frame: np.ndarray,
    template_name: str,
    *,
    threshold: float,
    debug_label: str | None = None,
) -> List[Tuple[int, int]]:
    template = _get_template(template_name)
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    if debug_label:
        print(f"[Vision] {debug_label} max={max_val:.3f} (threshold={threshold:.3f})")

    loc = np.where(result >= threshold)
    template_h, template_w = template.shape[:2]
    tol_x = max(4, template_w // 2)
    tol_y = max(4, template_h // 2)

    hits: List[Tuple[int, int]] = []
    for y, x in zip(loc[0], loc[1]):
        center = (int(x + template_w // 2), int(y + template_h // 2))
        if _is_new_center(center, hits, tol_x, tol_y):
            hits.append(center)
    return hits


def _ocr_find_keywords(
    frame: np.ndarray,
    keywords: Set[str],
    *,
    lang: str,
    debug_label: str | None = None,
    invert: bool = False,
) -> List[Tuple[int, int]]:
    if not ENABLE_OCR or pytesseract is None or TesseractOutput is None or not keywords:
        return []

    clean_keywords = {kw.strip() for kw in keywords if kw.strip()}
    if not clean_keywords:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(
        gray,
        None,
        fx=_OCR_SCALE,
        fy=_OCR_SCALE,
        interpolation=cv2.INTER_LINEAR,
    )
    if invert:
        scaled = cv2.bitwise_not(scaled)

    _, thresh = cv2.threshold(
        scaled,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    try:
        data = pytesseract.image_to_data(
            thresh,
            lang=lang,
            output_type=TesseractOutput.DICT,
        )
    except pytesseract.TesseractError as exc:  # pragma: no cover
        if debug_label:
            print(f"[OCR] {debug_label} failed: {exc}")
        return []

    results: List[Tuple[int, int]] = []
    best_conf = -1.0
    scale = _OCR_SCALE

    for i, text in enumerate(data.get("text", [])):
        clean = "".join(text.split())
        if not clean or clean not in clean_keywords:
            continue

        conf_str = data.get("conf", ["0"])[i]
        try:
            conf = float(conf_str)
        except ValueError:
            conf = -1.0
        best_conf = max(best_conf, conf)
        if conf < _OCR_MIN_CONF:
            continue

        left = data["left"][i]
        top = data["top"][i]
        width = data["width"][i]
        height = data["height"][i]

        center_x = int((left + width / 2) / scale)
        center_y = int((top + height / 2) / scale)
        results.append((center_x, center_y))

    if debug_label:
        print(f"[OCR] {debug_label} hits={len(results)} best_conf={best_conf:.1f}")

    return results


def _get_template(name: str) -> np.ndarray:
    cached = _TEMPLATE_CACHE.get(name)
    if cached is not None:
        return cached

    path = TEMPLATES_DIR / name
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Template not found: {path}")

    _TEMPLATE_CACHE[name] = image
    return image


def _is_new_center(
    candidate: Tuple[int, int],
    existing: Iterable[Tuple[int, int]],
    tol_x: int,
    tol_y: int,
) -> bool:
    cx, cy = candidate
    for ex, ey in existing:
        if abs(cx - ex) <= tol_x and abs(cy - ey) <= tol_y:
            return False
    return True


def _to_absolute(region: Dict[str, int], point: Tuple[int, int]) -> Tuple[int, int]:
    return (
        region["left"] + point[0],
        region["top"] + point[1],
    )

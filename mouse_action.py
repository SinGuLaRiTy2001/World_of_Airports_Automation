from __future__ import annotations

import ctypes
import time
from typing import Literal, Tuple


_user32 = ctypes.windll.user32
_GREEN = "\033[92m"
_RESET = "\033[0m"


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


_BUTTON_FLAG_MAP = {
    "left": (0x0002, 0x0004),   # down / up
    "right": (0x0008, 0x0010),
    "middle": (0x0020, 0x0040),
}


def get_position() -> Tuple[int, int]:
    """Return the cursor position in screen coordinates."""
    point = _POINT()
    _user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def move_to(x: int, y: int, duration: float = 0.0, steps: int = 24) -> None:
    """Move cursor to (x, y). Animate if duration > 0."""
    if duration <= 0.0 or steps <= 0:
        _user32.SetCursorPos(int(x), int(y))
        return

    start_x, start_y = get_position()
    steps = max(1, steps)
    delay = duration / steps

    for idx in range(1, steps + 1):
        nx = start_x + (x - start_x) * idx / steps
        ny = start_y + (y - start_y) * idx / steps
        _user32.SetCursorPos(int(round(nx)), int(round(ny)))
        time.sleep(delay)


def click(
    button: Literal["left", "right", "middle"] = "left",
    *,
    down_up_delay: float = 0.01,
    label: str | None = None,
) -> None:
    """Click the specified button at the current cursor position."""
    try:
        down_flag, up_flag = _BUTTON_FLAG_MAP[button]
    except KeyError as exc:
        raise ValueError(f"Unsupported button: {button}") from exc

    _user32.mouse_event(down_flag, 0, 0, 0, 0)
    if down_up_delay > 0.0:
        time.sleep(down_up_delay)
    _user32.mouse_event(up_flag, 0, 0, 0, 0)
    if label:
        print(f"{_GREEN}[Click]{_RESET} {label}")


def click_at(
    x: int,
    y: int,
    button: Literal["left", "right", "middle"] = "left",
    *,
    move_duration: float = 0.0,
    down_up_delay: float = 0.01,
    label: str | None = None,
) -> None:
    """Move to (x, y) and click."""
    move_to(x, y, move_duration)
    if label:
        print(f"{_GREEN}[Click]{_RESET} {label} @ ({x}, {y})")
    click(button, down_up_delay=down_up_delay)


def mouse_down(button: Literal["left", "right", "middle"] = "left") -> None:
    """Press and hold the specified mouse button."""
    try:
        down_flag, _ = _BUTTON_FLAG_MAP[button]
    except KeyError as exc:
        raise ValueError(f"Unsupported button: {button}") from exc
    _user32.mouse_event(down_flag, 0, 0, 0, 0)


def mouse_up(button: Literal["left", "right", "middle"] = "left") -> None:
    """Release the specified mouse button."""
    try:
        _, up_flag = _BUTTON_FLAG_MAP[button]
    except KeyError as exc:
        raise ValueError(f"Unsupported button: {button}") from exc
    _user32.mouse_event(up_flag, 0, 0, 0, 0)


def scroll(amount: int) -> None:
    """Scroll the mouse wheel by the given delta."""
    _user32.mouse_event(0x0800, 0, 0, int(amount), 0)


def scroll_down(steps: int = 1, *, step_delta: int = 120, pause: float = 0.05) -> None:
    """Scroll down in steps using the mouse wheel."""
    for _ in range(max(0, steps)):
        scroll(-abs(step_delta))
        if pause > 0.0:
            time.sleep(pause)


def scroll_up(steps: int = 1, *, step_delta: int = 120, pause: float = 0.05) -> None:
    """Scroll up in steps using the mouse wheel."""
    for _ in range(max(0, steps)):
        scroll(abs(step_delta))
        if pause > 0.0:
            time.sleep(pause)

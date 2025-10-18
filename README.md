# World of Airports Automation Bot

Automating the routine tasks in **World of Airports** can save a lot of time—especially when the right-hand event stack starts overflowing with attention icons. This project provides a Windows-based automation bot built with Python, OpenCV, and MSS that can keep your airport flowing while you focus on higher-level strategy (or grab a cup of coffee).

Of course, you can let most of the things handled by the tower in the game, using silver planes, but this one is FREE. :)

---

## Table of Contents

1. [Features](#features)
2. [How It Works](#how-it-works)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Configuration](#configuration)
8. [Extending the Bot](#extending-the-bot)
9. [Troubleshooting](#troubleshooting)

---

## Features

- **Attention Stack Automation**  
  Detects attention icons on the right-hand side of the game, scrolls to the bottom, and opens unresolved tasks in order.

- **Scenario-Specific Handlers**  
  Handles a variety of in-game dialogs:
  - Stand allocation (`SELECT_STAND_AVAILABLE` / `SELECT_STAND_CONFIRM`)
  - Ground crew assignment (drag anchor → start service)
  - Ground service “in-progress” tasks (mass-activate available buttons)
  - Completion dialogs, reward/claim buttons, pushback, runway crossing, taxi-to-hold, takeoff, and more.

- **Smart Card Selection**  
  Learns the baseline X-coordinate of untouched attention cards and skips clicking cards that are already open.

- **Responsive Hotkey**  
  Press `Ctrl + Space` to start/stop the automation loop. The bot waits 1 second after each handled scenario and honors all in-loop sleep durations.

- **Logging & Click Feedback**  
  Terminal output shows template/OCR confidence as well as green `[Click]` entries describing every automated click, helping you understand what the bot is doing in real time.

---

## How It Works

1. **Template Matching (OpenCV)**  
   The bot looks for specific UI elements using template matching. Each scenario has its own templates (PNG files) stored in `assets/templates/`.

2. **Scenario Dispatcher**  
   After selecting an attention card, the bot captures a screenshot and hands it to the scenario handlers. The first handler that matches takes over, performs its clicks, and the loop waits briefly before continuing.

3. **State Persistence**  
   The capture region (and the learned base X-coordinate of attention cards) is persisted in `woa_config.json`, so you only need to configure once.

4. **Mouse Automation**  
   Uses low-level Windows APIs (`ctypes.windll.user32`) to move, click, scroll, and drag the mouse.

---

## Requirements

- **Operating System**: Windows 10/11 (focus-based automation).  
- **Python**: 3.9 or newer (tested on 3.11/3.12).  
- **Packages**:
  - `opencv-python`
  - `numpy`
  - `mss`
  - `pynput`
  - (optional) `pytesseract` & Tesseract OCR for textual fallback

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Installation

1. **Clone / Copy the Repo**  
   Place the bot files in a dedicated folder, e.g. `WorldOfAirports_Automation`.

2. **Capture Region Setup**  
   - First run `init_window.py` directly (`python init_window.py`).  
   - Drag a rectangle covering your emulator/game window, press Enter to confirm.  
   - The capture region is saved to `woa_config.json`.

3. **Templates**  
   Ensure `assets/templates/` contains the PNGs used by each scenario (attention icon, select/confirm buttons, service icons, etc.).  
   If you notice missing matches, re-capture sharper templates from your own screen.

---

## Usage

1. Launch the game and make sure the capture region (window position/size) is the same as when you configured it.
   - During development and validation, the bot was run on a **2560×1440** Windows display, with *MuMuPlayer* (Android emulator) also set to **2560×1440** in maximized window mode.
   - All template assets were captured under that configuration and in Chinese. If you run the game with different resolutions or DPI scaling, you will need to recapture the templates (see [Extending the Bot](#extending-the-bot)).
2. Run the bot:

   ```bash
   python woa_bot.py
   ```

3. In the console, you’ll see `Press Ctrl + Enter to begin.`  
   Use `Ctrl + Space` to start or stop the automation loop at any time.
4. Watch the terminal output for `[Click]` messages and scenario reminders (e.g. `[Template] select_stand_title max=0.98 ...`).
5. When you’re done, hit `Ctrl + Space` to halt, then close the script with `Ctrl + C`.

**Tip**: Keep the emulator window in focus; the bot does not switch windows automatically.

---

## Project Structure

```
WorldOfAirports_Automation/
├── assets/
│   └── templates/           # PNG templates used by OpenCV
├── init_window.py           # Region selection & config persistence
├── mouse_action.py          # Mouse move/click/drag/scroll helpers
├── scenarios.py             # Scenario-specific detection & actions
├── woa_bot.py               # Main automation loop
├── woa_config.json          # Generated config (capture region, base X)
└── README.md                # Project documentation
```

---

## Extending the Bot

- **Add New Templates**  
  Drop additional PNGs into `assets/templates/` and reference them in `scenarios.py`.

- **Create a New Scenario**  
  Add a function in `scenarios.py` that detects and handles the target UI, such as upgrading planes and contracts.
  Register it in the `handlers` tuple inside `_process_active_scenarios` (order matters).

- **Tune Thresholds**  
  See `[Template] ... max=...` output to determine whether to tweak matching thresholds.

- **Enable OCR Fallback**  
  Set `ENABLE_OCR = True` in `scenarios.py` if you prefer textual detection when templates fail (requires Tesseract + language data).

- **Improve Responsiveness**  
  The current hotkey handling already checks between major steps. If you need even faster response, consider shorter sleep intervals or splitting long actions.

---

## Troubleshooting

| Symptom                                    | Possible Fix                                                                 |
|--------------------------------------------|-------------------------------------------------------------------------------|
| No `[Click]` even when `[Template]` is high | The matching step moves on to the next element (e.g., `stand_confirm`). Ensure both templates exist or adjust thresholds. |
| “Missing template” messages                 | The script couldn’t find the PNG referenced; verify file names/locations.     |
| OCR fallback never fires                    | Install Tesseract and set `ENABLE_OCR = True`, or capture cleaner templates.  |
| Hotkey slow to stop                        | The loop waits for current action to finish; if necessary, break long sleeps into shorter intervals. |

---

## Contributing & License

Feel free to fork this project, add new features, or refine the existing scenarios. This repository is shared under the **MIT License**. Contributions via pull requests are welcome—please include screenshots or logs demonstrating your changes.

---

Enjoy keeping your airport running smoothly with minimal manual clicks!  
If you have issues or ideas for new scenarios, open an issue or start a discussion.  
Happy flying! ✈️

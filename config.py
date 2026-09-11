# Auth & Session Defaults (Local-friendly defaults so run_local works with zero config)
from datetime import datetime, timezone
from pathlib import Path
import re
from queue import Empty, Full
from typing import Literal, cast
import os
from dotenv import load_dotenv

BASE_DIR: Path = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
LOCAL_ADMIN_EDIT_PASSWORD = "local-ocr-admin"
LOCAL_SESSION_SECRET = "local-ocr-session-secret-change-me"
ADMIN_EDIT_PASSWORD = os.getenv("ADMIN_EDIT_PASSWORD", LOCAL_ADMIN_EDIT_PASSWORD)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
SESSION_SECRET = os.getenv("SESSION_SECRET", LOCAL_SESSION_SECRET)
SESSION_COOKIE_NAME = "ocr_admin_session"
SESSION_LIFETIME_SECONDS = 60 * 60 * 8
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"
cookie_samesite = os.getenv("COOKIE_SAMESITE", "lax").lower()
if cookie_samesite not in {"lax", "strict", "none"}:
    raise RuntimeError("COOKIE_SAMESITE must be lax, strict, or none")
COOKIE_SAMESITE: Literal["lax", "strict", "none"] = cast(
    Literal["lax", "strict", "none"], cookie_samesite
)

# Server & Stream Settings
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173").split(",")
DEFAULT_STREAM_URL: str | None = os.getenv("STREAM_URL")
SAMPLE_INTERVAL_SECONDS: float = 2.0
PREFERRED_QUALITIES = ("720p", "720p60", "480p", "480p60", "best")
DEBUG_OCR: bool = os.getenv("DEBUG_OCR", "false").lower() == "true"

# Asset Paths & OCR Config
ASSET_DIRECTORY: Path = BASE_DIR.parent / "test-assets"
ASSET_DIRECTORY.mkdir(parents=True, exist_ok=True)
IMAGES_DIR: Path = BASE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TESSDATA_DIR = BASE_DIR
CUSTOM_DIGITS_PATH = TESSDATA_DIR / "custom_digits.traineddata"
TESSERACT_CONFIG = f'--tessdata-dir "{TESSDATA_DIR}" -l custom_digits --psm 7 -c tessedit_char_whitelist=0123456789:+'

# OCR & Game Constants
BLUE_ROTATION: float = -4.0
ORANGE_ROTATION: float = 4.0
BOOTSTRAP_CONFIRMATIONS: int = 3
ANOMALY_CONFIRMATIONS: int = 2
SAMPLE_CLOCK_TOLERANCE_SECONDS: int = 5
MAX_SCORE: int = 20
TIME_PATTERN = re.compile(r"\+?(\d{1,2}):(\d{2})")
TIME_DIGITS_PATTERN = re.compile(r"\+?(\d{3,4})")

def publish_game_state_event(
    status: Literal["queued", "running", "completed", "failed", "idle", "stopping"],
    message: str,
    game_state=None,
) -> None:
    """Store and broadcast a game-state lifecycle event to SSE subscribers."""
    import constants
    from models import GameStateEvent

    with constants.event_lock:
        event = GameStateEvent(
            id=constants.next_event_id,
            timestamp=datetime.now(timezone.utc),
            status=status,
            message=message,
            game_state=game_state,
        )
        constants.next_event_id += 1
        constants.latest_event = event
        event_json = event.model_dump_json(exclude_none=True)
        subscribers = tuple(constants.event_subscribers)

    for subscriber in subscribers:
        try:
            subscriber.put_nowait(event_json)
        except Full:
            try:
                subscriber.get_nowait()
            except Empty:
                pass
            try:
                subscriber.put_nowait(event_json)
            except Full:
                pass
from collections import Counter
import os
import sys
import time
import types
from typing import Any, Literal, Sequence
import cv2
from cv2.typing import MatLike
from fastapi import HTTPException
import numpy as np
import pytesseract

from auth import decode_base64, encode_base64
from config import (
    ANOMALY_CONFIRMATIONS,
    BLUE_ROTATION,
    IMAGES_DIR,
    MAX_SCORE,
    ORANGE_ROTATION,
    SAMPLE_CLOCK_TOLERANCE_SECONDS,
    TESSERACT_CONFIG,
    TIME_DIGITS_PATTERN,
    TIME_PATTERN,
)
import constants
from models import (
    GameState,
    GameStateEvent,
    GameTimeState,
    NormalizedRectangle,
    OCRCalibration,
    PreviewFrameMetadata,
    ScoreState,
)

# Aliases for backwards compatibility with tests and older imports
video_state = constants.video_state
stop_requested = constants.stop_requested
refresh_requested = constants.refresh_requested
preview_requested = constants.preview_requested
preview_lock = constants.preview_lock
calibration_lock = constants.calibration_lock
event_lock = constants.event_lock

# Backwards compatible encode/decode functions
_encode_base64 = encode_base64
_decode_base64 = decode_base64


def current_calibration() -> OCRCalibration:
    """Return an isolated snapshot of the active OCR calibration."""
    with constants.calibration_lock:
        return constants.active_calibration.model_copy(deep=True)

def normalized_crop(frame: np.ndarray, rectangle: NormalizedRectangle) -> np.ndarray:
    """Extract a bounded pixel crop from normalized frame coordinates."""
    frame_height, frame_width = frame.shape[:2]
    x1 = int(rectangle.x * frame_width)
    y1 = int(rectangle.y * frame_height)
    x2 = max(x1 + 1, min(frame_width, int((rectangle.x + rectangle.width) * frame_width)))
    y2 = max(y1 + 1, min(frame_height, int((rectangle.y + rectangle.height) * frame_height)))
    return frame[y1:y2, x1:x2]


def encode_preview_jpeg(frame: np.ndarray) -> bytes:
    """Downscale a frame when needed and encode it as a JPEG preview."""
    preview = frame
    max_preview_width = 1280
    if preview.shape[1] > max_preview_width:
        scale = max_preview_width / preview.shape[1]
        preview = cv2.resize(preview, (max_preview_width, int(preview.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    encoded, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not encoded:
        raise RuntimeError("Unable to encode the preview frame")
    return jpeg.tobytes()


def threshold_preview_frame(frame: np.ndarray, calibration: OCRCalibration) -> np.ndarray:
    """Render calibration-aligned OCR threshold masks on a black canvas."""
    threshold_preview = np.zeros_like(frame)
    regions = (
        (calibration.blue_score, "blue_score"),
        (calibration.time, "time"),
        (calibration.orange_score, "orange_score"),
    )

    frame_height, frame_width = frame.shape[:2]
    for rectangle, zone_name in regions:
        x1 = int(rectangle.x * frame_width)
        y1 = int(rectangle.y * frame_height)
        x2 = max(x1 + 1, min(frame_width, int((rectangle.x + rectangle.width) * frame_width)))
        y2 = max(y1 + 1, min(frame_height, int((rectangle.y + rectangle.height) * frame_height)))
        crop = frame[y1:y2, x1:x2]
        if zone_name.endswith("score"):
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([0, 0, 180], dtype=np.uint8)
            upper_bound = np.array([179, 100, 255], dtype=np.uint8)
            threshold = cv2.inRange(hsv, lower_bound, upper_bound)
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blurred_gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, threshold = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_preview[y1:y2, x1:x2] = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)

    return threshold_preview 


def cache_preview_frame(frame: np.ndarray) -> None:
    """Cache the latest source frame, JPEG preview, and capture metadata."""
    frame_height, frame_width = frame.shape[:2]
    preview_jpeg = encode_preview_jpeg(frame)

    with constants.preview_lock:
        constants.latest_preview_frame = frame.copy()
        constants.latest_preview_jpeg = preview_jpeg
        constants.latest_preview_metadata = PreviewFrameMetadata(
            capture_id=constants.next_preview_capture_id,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        constants.next_preview_capture_id += 1

    # Write latest normal vision and threshold vision frames to images directory
    write_image(frame, "preview_normal")
    write_image(threshold_preview_frame(frame, current_calibration()), "preview_threshold")


def format_sse_event(event_json: str) -> str:
    """Format serialized event data for a Server-Sent Events response."""
    return f"event: game_state\ndata: {event_json}\n\n"


def rotate_image_capture_bound_box(image, angle) -> MatLike:
    """Rotate an OCR crop while preserving its original dimensions."""
    if angle == 0.0:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

def preprocess_with_grayscale_kernel_thresh_and_padding(coordinates_img, zone_name: str) -> MatLike:
    """Prepare a score or timer crop for Tesseract threshold-based OCR."""
    if zone_name.endswith("score"):
        hsv = cv2.cvtColor(coordinates_img, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 0, 180], dtype=np.uint8)
        upper_bound = np.array([179, 100, 255], dtype=np.uint8)
        thresh = cv2.inRange(hsv, lower_bound, upper_bound)
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    else:
        gray = cv2.cvtColor(coordinates_img, cv2.COLOR_BGR2GRAY)
        enlarged_gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        blurred_gray = cv2.GaussianBlur(enlarged_gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    padded_thresh = cv2.copyMakeBorder(
        thresh,
        top=8,
        bottom=8,
        left=8,
        right=8,
        borderType=cv2.BORDER_CONSTANT,
        value=(0,),
    )
    return padded_thresh


def write_image(coordinates_img: np.ndarray | None, zone_name: str) -> None:
    """Write an OCR debug/preview image to the backend/images directory."""
    if coordinates_img is None:
        return
    try:
        clean_name = zone_name.replace("-", "_")
        output_path: str = str(IMAGES_DIR / f"{clean_name}.png")
        cv2.imwrite(output_path, coordinates_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    except Exception as exc:
        print(f"Failed to write image {zone_name}: {exc}")

def process_similar_numbers(detected_num: str | None, corrected_canvas: np.ndarray) -> str | Literal["N/A"]:
    """Apply shape heuristics to distinguish commonly confused OCR digits."""
    if detected_num in ["5", "4", "7"]:
        upscaled_thresh: np.ndarray = cv2.resize(corrected_canvas, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        contours, _ = cv2.findContours(upscaled_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour: np.ndarray = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            digit_crop: np.ndarray = upscaled_thresh[y : y + h, x : x + w]
            aspect_ratio: float = w / float(h)
            if detected_num == "4" and aspect_ratio < 0.45:
                return "1"
            if detected_num == "5":
                half_h = h // 2
                top_half_density = cv2.countNonZero(digit_crop[0:half_h, :])
                bottom_half_density = cv2.countNonZero(digit_crop[half_h:h, :])
                if top_half_density > bottom_half_density * 1.25:
                    return "7"
    return detected_num if detected_num else "N/A"


def parse_time_string(time_str: str) -> tuple[int, int, bool] | None:
    """Parse a raw OCR clock string adhering to Rocket League regulation and overtime rules."""
    time_str = time_str.strip()
    is_overtime = False

    if time_str.startswith("+"):
        is_overtime = True
        time_str = time_str[1:]

    parts = time_str.split(":")
    if len(parts) != 2:
        return None

    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
    except ValueError:
        return None

    if not (0 <= seconds < 60):
        return None

    # Rocket League regulation matches are 5 minutes maximum (5:00).
    # In Overtime, '+' is frequently recognized by OCR as a leading '1' digit (e.g. '+0:34' -> '10:34', '+1:15' -> '11:15').
    if not is_overtime and minutes >= 10:
        is_overtime = True
        minutes -= 10

    if minutes > 30:
        return None

    return minutes, seconds, is_overtime


def determine_number(coordinates_img: np.ndarray, zone_name: str) -> str | Literal["N/A"]:
    """Read and validate one score or clock value from a prepared crop."""
    padded_thresh: np.ndarray = preprocess_with_grayscale_kernel_thresh_and_padding(coordinates_img, zone_name)
    if zone_name.startswith("blue"):
        corrected_canvas = rotate_image_capture_bound_box(padded_thresh, BLUE_ROTATION)
    elif zone_name.startswith("orange"):
        corrected_canvas = rotate_image_capture_bound_box(padded_thresh, ORANGE_ROTATION)
    else:
        corrected_canvas = padded_thresh

    clean_zone = zone_name.replace("-", "_")
    write_image(corrected_canvas, f"{clean_zone}_thresh")
    if zone_name.endswith("score"):
        contours: Sequence[MatLike]
        contours, _ = cv2.findContours(corrected_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour: np.ndarray = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 15:
                _, _, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio: float = w / float(h)
                if aspect_ratio < 0.38:
                    return "1"

    detected_num: str = pytesseract.image_to_string(corrected_canvas, config=TESSERACT_CONFIG).strip()
    result: str = process_similar_numbers(detected_num, corrected_canvas)
    if zone_name == "time":
        if TIME_PATTERN.fullmatch(result):
            return result
        if TIME_DIGITS_PATTERN.fullmatch(result):
            digits = result.lstrip("+")
            prefix = "+" if result.startswith("+") else ""
            if len(digits) == 3:
                return f"{prefix}{digits[0]}:{digits[1:]}"
            if len(digits) == 4:
                return f"{prefix}{digits[:2]}:{digits[2:]}"
        return "N/A"
    if result.isdigit() and 0 <= int(result) <= MAX_SCORE:
        return result
    return "N/A"


def get_coordinates(original_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract calibrated blue-score, orange-score, and timer image regions."""
    calibration = current_calibration()
    write_image(original_img, "preview_normal")
    write_image(threshold_preview_frame(original_img, calibration), "preview_threshold")
    blue_coordinates = normalized_crop(original_img, calibration.blue_score)
    orange_coordinates = normalized_crop(original_img, calibration.orange_score)
    time_coordinates = normalized_crop(original_img, calibration.time)
    write_image(blue_coordinates, "blue_score_raw")
    write_image(orange_coordinates, "orange_score_raw")
    write_image(time_coordinates, "time_raw")
    return blue_coordinates, orange_coordinates, time_coordinates

def get_score(
    original_img: np.ndarray,
    blue_coordinates: np.ndarray,
    orange_coordinates: np.ndarray,
    time_coordinates: np.ndarray,
) -> tuple[ScoreState, ScoreState, GameTimeState]:
    """Run OCR for both scores and the game clock from image crops."""
    if original_img is None:
        return ScoreState(blue_score=0, orange_score=0), ScoreState(blue_score=0, orange_score=0), GameTimeState()
    blue_score_str: str = determine_number(blue_coordinates, "blue-score")
    orange_score_str: str = determine_number(orange_coordinates, "orange-score")
    time_left_str: str = determine_number(time_coordinates, "time")

    blue_score = int(blue_score_str) if blue_score_str.isdigit() and int(blue_score_str) <= MAX_SCORE else None
    orange_score = int(orange_score_str) if orange_score_str.isdigit() and int(orange_score_str) <= MAX_SCORE else None

    parsed_time = parse_time_string(time_left_str)
    if parsed_time is not None:
        minutes, seconds, is_overtime = parsed_time
        time_left = GameTimeState(time_minutes=minutes, time_seconds=seconds, is_overtime=is_overtime)
    else:
        time_left = GameTimeState()

    score_state = ScoreState(blue_score=blue_score, orange_score=orange_score)
    return score_state, score_state, time_left


def format_time_data(time_left: GameTimeState) -> tuple[int | None | Literal["xx"], int | None | Literal["xx"], bool]:
    """Convert a game-clock model into its minute, second, and overtime values."""
    return time_left.time_minutes, time_left.time_seconds, time_left.is_overtime


def parse_detected_time(time_data: tuple | list) -> tuple[int, int, bool] | None:
    """Return a valid numeric clock tuple or discard an incomplete reading."""
    minutes = time_data[0]
    seconds = time_data[1]
    is_overtime = bool(time_data[2]) if len(time_data) > 2 else False
    if isinstance(minutes, int) and isinstance(seconds, int) and 0 <= minutes < 100 and 0 <= seconds < 60:
        return minutes, seconds, is_overtime
    return None


def time_to_seconds(time_data: tuple[int, int, bool] | None) -> tuple[int, bool] | None:
    """Convert a clock tuple into total seconds and overtime status."""
    if time_data is None:
        return None
    minutes, seconds, is_overtime = time_data
    return minutes * 60 + seconds, is_overtime

def seconds_to_game_time(clock_info: tuple[int, bool] | int | None) -> GameTimeState:
    """Convert total seconds into the public game-clock model."""
    if clock_info is None:
        return GameTimeState()
    if isinstance(clock_info, tuple):
        seconds, is_overtime = clock_info
    else:
        seconds, is_overtime = clock_info, False
    return GameTimeState(time_minutes=seconds // 60, time_seconds=seconds % 60, is_overtime=is_overtime)

class ClockTracker:
    """Track and stabilize the game clock using monotonic elapsed time and Rocket League rules."""

    def __init__(self) -> None:
        self.clock_seconds: int | None = None
        self.is_overtime: bool = False
        self.last_update_monotonic: float | None = None
        self.pending_sample: tuple[int, bool, float] | None = None

    def update(self, detected_seconds: int | None, is_overtime: bool) -> tuple[int | None, bool]:
        """Incorporate one clock sample and return the stabilized clock seconds and overtime state."""
        now = time.monotonic()
        if detected_seconds is None:
            return self.clock_seconds, self.is_overtime

        # 1. First sample: adopt immediately on Sample 1
        if self.clock_seconds is None or self.last_update_monotonic is None:
            self.clock_seconds = detected_seconds
            self.is_overtime = is_overtime
            self.last_update_monotonic = now
            self.pending_sample = None
            return self.clock_seconds, self.is_overtime

        elapsed_wall = max(0.0, now - self.last_update_monotonic)

        # If the active clock is stale (>15s without a consistent reading), adopt immediately
        if elapsed_wall > 15.0:
            self.clock_seconds = detected_seconds
            self.is_overtime = is_overtime
            self.last_update_monotonic = now
            self.pending_sample = None
            return self.clock_seconds, self.is_overtime

        # 2. Check if reading is consistent with our active clock progression
        if self._is_consistent(self.clock_seconds, self.is_overtime, detected_seconds, is_overtime, elapsed_wall):
            self.clock_seconds = detected_seconds
            self.is_overtime = is_overtime
            self.last_update_monotonic = now
            self.pending_sample = None
            return self.clock_seconds, self.is_overtime

        # 3. Reading differs from active clock (could be a one-frame glitch or a real jump)
        if self.pending_sample is not None:
            pend_sec, pend_ot, pend_time = self.pending_sample
            pend_elapsed = max(0.0, now - pend_time)
            if pend_elapsed > 10.0:
                self.pending_sample = None
            # If confirmed by two consecutive samples, adopt the new clock baseline
            elif self._is_consistent(pend_sec, pend_ot, detected_seconds, is_overtime, pend_elapsed):
                self.clock_seconds = detected_seconds
                self.is_overtime = is_overtime
                self.last_update_monotonic = now
                self.pending_sample = None
                return self.clock_seconds, self.is_overtime

        # Store as candidate in pending; active clock remains protected
        self.pending_sample = (detected_seconds, is_overtime, now)
        return self.clock_seconds, self.is_overtime

    def _is_consistent(self, base_sec: int, base_ot: bool, new_sec: int, new_ot: bool, elapsed_wall: float) -> bool:
        """Check if new_sec is a plausible continuation from base_sec after elapsed_wall seconds."""
        if base_ot != new_ot:
            if not base_ot and new_ot and base_sec <= 5 and new_sec <= 5:
                return True
            return False

        # Tolerance allows for normal video frame timing variations and goal replay pauses (delta = 0)
        tolerance = max(float(SAMPLE_CLOCK_TOLERANCE_SECONDS), elapsed_wall + 3.0)
        if base_ot:
            # Overtime: clock counts UP
            delta = new_sec - base_sec
            return 0 <= delta <= tolerance
        else:
            # Regulation: clock counts DOWN
            delta = base_sec - new_sec
            return 0 <= delta <= tolerance

class ScoreTracker:
    """Track and stabilize blue and orange scores adhering to Rocket League score rules."""

    def __init__(self) -> None:
        self.blue_score: int | None = None
        self.orange_score: int | None = None
        self.pending_score: tuple[int | None, int | None] | None = None
        self.pending_count: int = 0

    def update(self, blue: int | None, orange: int | None) -> tuple[int | None, int | None]:
        """Incorporate score readings, filtering out transient OCR noise."""
        if blue is None and orange is None:
            return self.blue_score, self.orange_score

        # First score initialization: adopt on first valid sample
        if self.blue_score is None and blue is not None:
            self.blue_score = blue
        if self.orange_score is None and orange is not None:
            self.orange_score = orange

        if blue is None or orange is None or self.blue_score is None or self.orange_score is None:
            return self.blue_score, self.orange_score

        # Normal score progression: stays same or increments by 1
        blue_valid = blue in (self.blue_score, self.blue_score + 1)
        orange_valid = orange in (self.orange_score, self.orange_score + 1)

        if blue_valid and orange_valid:
            self.blue_score = blue
            self.orange_score = orange
            self.pending_score = None
            self.pending_count = 0
            return self.blue_score, self.orange_score

        # Score anomaly / jump: require 2 confirmations before adopting
        if self.pending_score == (blue, orange):
            self.pending_count += 1
            if self.pending_count >= ANOMALY_CONFIRMATIONS:
                self.blue_score = blue
                self.orange_score = orange
                self.pending_score = None
                self.pending_count = 0
        else:
            self.pending_score = (blue, orange)
            self.pending_count = 1

        return self.blue_score, self.orange_score

class GameStateReducer:
    """Stabilize noisy OCR samples into a plausible Rocket League game state."""

    def __init__(self) -> None:
        self.score_tracker = ScoreTracker()
        self.clock_tracker = ClockTracker()
        self.winner: Literal["Blue", "Orange"] | None = None
        self.is_game_over: bool = False
        self.game_over_sample_count: int = 0
        self.low_time_first_seen: float | None = None

    def reset(self) -> None:
        """Reset all tracked match state for a fresh game."""
        self.score_tracker = ScoreTracker()
        self.clock_tracker = ClockTracker()
        self.winner = None
        self.is_game_over = False
        self.game_over_sample_count = 0
        self.low_time_first_seen = None

    def update(
        self,
        blue_score: int | None,
        orange_score: int | None,
        clock_input: tuple[int, bool] | int | None,
    ) -> GameState:
        """Incorporate one OCR observation and return the stabilized state."""
        now = time.monotonic()

        if isinstance(clock_input, tuple):
            clock_sec, is_ot = clock_input
        elif isinstance(clock_input, int):
            clock_sec, is_ot = clock_input, False
        else:
            clock_sec, is_ot = None, False

        # If previous game ended, check if a new game started to trigger auto-reset
        if self.is_game_over:
            self.game_over_sample_count += 1
            new_game_clock = clock_sec is not None and not is_ot and clock_sec >= 270  # >= 4:30
            new_game_score = (blue_score == 0 and orange_score == 0)
            if new_game_clock or (self.game_over_sample_count >= 4 and
                (new_game_score or (blue_score is not None and orange_score is not None and blue_score != self.score_tracker.blue_score))):
                self.reset()

        b, o = self.score_tracker.update(blue_score, orange_score)
        sec, ot = self.clock_tracker.update(clock_sec, is_ot)

        # Track low-time (< 10s) duration in regulation
        if not ot and sec is not None and sec <= 10:
            if self.low_time_first_seen is None:
                self.low_time_first_seen = now
        elif sec is not None and sec > 10:
            self.low_time_first_seen = None

        # Check for Rocket League Game Over / Winner conditions:
        if not self.is_game_over and b is not None and o is not None:
            # 1. Regulation End: Clock reaches 0:00 (or <= 0:01) with non-tied scores
            if not ot and sec is not None and sec <= 1 and b != o:
                self.is_game_over = True
                self.winner = "Blue" if b > o else "Orange"
            # 2. Low-time condition: Clock under 10 seconds for more than 30 seconds
            elif not ot and self.low_time_first_seen is not None and (now - self.low_time_first_seen >= 30.0) and b != o:
                self.is_game_over = True
                self.winner = "Blue" if b > o else "Orange"
            # 3. Overtime End (Sudden Death): Overtime mode and one team leads
            elif ot and b != o:
                self.is_game_over = True
                self.winner = "Blue" if b > o else "Orange"

        return GameState(
            score_state=ScoreState(blue_score=b, orange_score=o),
            time_left=seconds_to_game_time((sec, ot) if sec is not None else None),
            winner=self.winner,
            is_game_over=self.is_game_over,
        )

    @property
    def blue_score(self) -> int | None:
        return self.score_tracker.blue_score

    @property
    def orange_score(self) -> int | None:
        return self.score_tracker.orange_score

    @property
    def clock_seconds(self) -> int | None:
        return self.clock_tracker.clock_seconds

    @property
    def is_overtime(self) -> bool:
        return self.clock_tracker.is_overtime

def most_frequent_scores_with_recent_tiebreak(values: Sequence[int]) -> int:
    """Choose the most frequent score, preferring the latest tied value."""
    counts = Counter(values)
    max_count: int = max(counts.values())
    candidates: set[int | Literal["N/A"] | None] = {value for value, count in counts.items() if count == max_count}

    for value in reversed(values):
        if value in candidates:
            return value

    return values[-1]

def get_ocr_result(
    frame: Any,
) -> tuple[int | None, int | None, tuple[int | None | Literal["xx"], int | None | Literal["xx"], bool]]:
    """Extract raw blue score, orange score, and clock readings from a frame."""
    blue_coordinates, orange_coordinates, time_coordinates = get_coordinates(frame)
    blue_score, orange_score, time_left = get_score(frame, blue_coordinates, orange_coordinates, time_coordinates)
    formatted_time = format_time_data(time_left)
    blue_score_result = blue_score.blue_score if isinstance(blue_score.blue_score, int) else None
    orange_score_result = orange_score.orange_score if isinstance(orange_score.orange_score, int) else None
    return blue_score_result, orange_score_result, formatted_time


def resolve_stream_url(stream_url: str) -> str:
    """Resolve a stream URL or local path to a compatible playable stream URL."""
    if not stream_url or not stream_url.strip():
        raise HTTPException(status_code=400, detail="No stream URL provided")

    cleaned_url = stream_url.strip()

    # Local file path, webcam index, or direct stream video format
    if (
        os.path.exists(cleaned_url)
        or cleaned_url.isdigit()
        or ".m3u8" in cleaned_url
        or any(cleaned_url.lower().endswith(ext) for ext in (".mp4", ".mkv", ".mov", ".ts", ".flv", ".webm", ".avi"))
    ):
        return cleaned_url

    try:
        from streamlink.session.session import Streamlink

        streams = Streamlink().streams(cleaned_url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to resolve stream (channel may be offline or invalid): {exc}") from None

    for quality in ("480p", "480p60", "720p", "720p60", "best"):
        stream = streams.get(quality)
        if stream:
            return stream.to_url()
    raise HTTPException(status_code=400, detail="No compatible stream quality is available for this channel")

class _OCRModule(types.ModuleType):
    @property
    def app(self):
        import endpoints
        return endpoints.app

    @property
    def active_calibration(self) -> OCRCalibration:
        return constants.active_calibration

    @active_calibration.setter
    def active_calibration(self, value: OCRCalibration) -> None:
        with constants.calibration_lock:
            constants.active_calibration = value

    @property
    def latest_event(self) -> GameStateEvent | None:
        return constants.latest_event

    @latest_event.setter
    def latest_event(self, value: GameStateEvent | None) -> None:
        with constants.event_lock:
            constants.latest_event = value

    @property
    def active_stream_url(self) -> str | None:
        return constants.active_stream_url

    @active_stream_url.setter
    def active_stream_url(self, value: str | None) -> None:
        constants.active_stream_url = value

    @property
    def latest_preview_jpeg(self) -> bytes | None:
        return constants.latest_preview_jpeg

    @latest_preview_jpeg.setter
    def latest_preview_jpeg(self, value: bytes | None) -> None:
        constants.latest_preview_jpeg = value

    @property
    def latest_preview_frame(self) -> np.ndarray | None:
        return constants.latest_preview_frame

    @latest_preview_frame.setter
    def latest_preview_frame(self, value: np.ndarray | None) -> None:
        constants.latest_preview_frame = value

    @property
    def latest_preview_metadata(self) -> PreviewFrameMetadata | None:
        return constants.latest_preview_metadata

    @latest_preview_metadata.setter
    def latest_preview_metadata(self, value: PreviewFrameMetadata | None) -> None:
        constants.latest_preview_metadata = value


sys.modules[__name__].__class__ = _OCRModule
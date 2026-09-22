import threading
from queue import Queue
from typing import Any
import numpy as np
from models import (
    GameStateEvent,
    NormalizedRectangle,
    OCRCalibration,
    PreviewFrameMetadata,
    VideoState,
)

# Locks and thread synchronization events
video_state_lock = threading.Lock()
event_lock = threading.RLock()
refresh_request_lock = threading.Lock()
refresh_requested = threading.Event()
refresh_completed = threading.Event()
preview_request_lock = threading.Lock()
preview_requested = threading.Event()
preview_completed = threading.Event()
reducer_reset_requested = threading.Event()
stop_requested = threading.Event()
calibration_lock = threading.Lock()
preview_lock = threading.Lock()

# Shared runtime state
event_subscribers: set[Queue[str]] = set()
latest_event: GameStateEvent | None = None
next_event_id: int = 1

active_calibration = OCRCalibration(
    blue_score=NormalizedRectangle(x=0.415, y=0.0, width=0.025, height=0.085),
    time=NormalizedRectangle(x=0.460, y=0.0, width=0.080, height=0.105),
    orange_score=NormalizedRectangle(x=0.555, y=0.0, width=0.035, height=0.085),
)

latest_preview_jpeg: bytes | None = None
latest_preview_frame: np.ndarray | None = None
latest_preview_metadata: PreviewFrameMetadata | None = None
next_preview_capture_id: int = 1
active_stream_url: str | None = None

video_state = VideoState()
active_live_capture: Any = None

# Public viewer controls lock
controls_locked: bool = False

import asyncio
import hmac
import os
from queue import Empty, Queue
import threading
import time
import traceback
from typing import Literal, cast
import cv2
from fastapi import BackgroundTasks, Cookie, Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import numpy as np
from numpy.typing import NDArray

from auth import create_admin_session_token, is_valid_admin_credentials, require_admin_session
from config import (
    COOKIE_SAMESITE,
    COOKIE_SECURE,
    FRONTEND_ORIGINS,
    SESSION_COOKIE_NAME,
    SESSION_LIFETIME_SECONDS,
    SESSION_SECRET,
    publish_game_state_event,
)
import constants
from datetime import datetime, timezone
from models import (
    AdminPasswordRequest,
    AdminSessionState,
    GameState,
    GameStateEvent,
    GameTimeState,
    NormalizedRectangle,
    OCRCalibration,
    OCRRegions,
    PreviewFrameMetadata,
    ScoreState,
    StreamStartRequest,
    StreamUrlRequest,
    VideoJobStatus,
    VideoState,
)
import ocr

app: FastAPI = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in FRONTEND_ORIGINS if origin.strip()],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.middleware("http")
async def normalize_slashes_middleware(request: Request, call_next):
    path = request.scope.get("path", "")
    if "//" in path:
        import re
        request.scope["path"] = re.sub(r"/+", "/", path)
    return await call_next(request)


@app.post("/refresh-game-state")
def refresh_game_state() -> GameState:
    """Request and return one immediate OCR state sample from the active job."""
    if constants.video_state.status != "running":
        raise HTTPException(status_code=409, detail="No active OCR video job is available to refresh")
    if not constants.refresh_request_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A game-state refresh is already in progress")

    try:
        constants.refresh_completed.clear()
        constants.refresh_requested.set()
        if not constants.refresh_completed.wait(timeout=6):
            raise HTTPException(status_code=504, detail="Timed out waiting for an OCR refresh")
        if constants.video_state.status != "running":
            raise HTTPException(status_code=409, detail="The OCR video job stopped before it could refresh")
        with constants.event_lock:
            game_state = constants.latest_event.game_state if constants.latest_event else None
            return game_state or GameState(score_state=ScoreState(), time_left=GameTimeState())
    finally:
        constants.refresh_request_lock.release()


@app.post("/stop-local-video")
def stop_local_video() -> VideoState:
    """Ask the active OCR capture loop to stop and return the updated video state."""
    if constants.video_state.status not in {"queued", "running", "stopping"}:
        raise HTTPException(status_code=409, detail="No OCR video job is running")
    constants.stop_requested.set()
    constants.video_state.status = "stopping"
    constants.video_state.message = "OCR video processing stop requested"
    publish_game_state_event("stopping", "OCR video stop requested")

    # Wait briefly for capture loop to exit and transition to idle
    t_end = time.monotonic() + 3.0
    while time.monotonic() < t_end and constants.video_state.status != "idle":
        time.sleep(0.05)

    return constants.video_state


@app.get("/video-state")
def get_video_state() -> VideoState:
    """Return the current OCR video job status and configuration."""
    constants.video_state.message = f"OCR video status is {constants.video_state.status}"
    return constants.video_state


@app.get("/ocr-calibration")
def get_ocr_calibration(_: None = Depends(require_admin_session)) -> OCRCalibration:
    """Return the authenticated editor's current OCR calibration."""
    return ocr.current_calibration()


@app.get("/ocr-regions")
def get_ocr_regions() -> OCRRegions:
    """Return current normalized OCR bounding box regions formatted with timer key."""
    cal = ocr.current_calibration()
    return OCRRegions(
        blue_score=cal.blue_score,
        timer=cal.time,
        orange_score=cal.orange_score,
    )


@app.put("/ocr-regions")
def update_ocr_regions(regions: OCRRegions, _: None = Depends(require_admin_session)) -> OCRRegions:
    """Save calibration regions atomically and restart state bootstrap when active."""
    with constants.calibration_lock:
        constants.active_calibration = OCRCalibration(
            blue_score=regions.blue_score,
            time=regions.timer,
            orange_score=regions.orange_score,
        )
        constants.reducer_reset_requested.set()
        if constants.video_state.status == "running":
            with constants.event_lock:
                game_state = constants.latest_event.game_state if constants.latest_event else None
            publish_game_state_event("running", "OCR calibration updated; state bootstrap restarted", game_state)
    cal = ocr.current_calibration()
    return OCRRegions(
        blue_score=cal.blue_score,
        timer=cal.time,
        orange_score=cal.orange_score,
    )


@app.put("/ocr-calibration")
def update_ocr_calibration(calibration: OCRCalibration, _: None = Depends(require_admin_session)) -> OCRCalibration:
    """Save calibration atomically and restart state bootstrap when active."""
    with constants.calibration_lock:
        constants.active_calibration = calibration.model_copy(deep=True)
        constants.reducer_reset_requested.set()
        if constants.video_state.status == "running":
            with constants.event_lock:
                game_state = constants.latest_event.game_state if constants.latest_event else None
            publish_game_state_event("running", "OCR calibration updated; state bootstrap restarted", game_state)
    return ocr.current_calibration()


@app.put("/stream-url")
def update_stream_url(payload: StreamUrlRequest) -> dict[str, str]:
    """Update the active stream/video URL submitted from the frontend."""
    if payload.stream_url and payload.stream_url.strip():
        constants.active_stream_url = payload.stream_url.strip()
    return {
        "stream_url": str(constants.active_stream_url or ""),
        "message": "Stream URL updated successfully",
    }

@app.post("/start-local-video")
async def start_local_video(
    background_tasks: BackgroundTasks,
    start_request: StreamStartRequest | None = None,
) -> VideoState:
    """Start local video OCR with the submitted stream URL (alias for /run-local-video)."""
    return await trigger_run_local_video(background_tasks, start_request or StreamStartRequest())


@app.post("/refresh-preview-frame")
def refresh_preview_frame(_: None = Depends(require_admin_session)) -> PreviewFrameMetadata:
    """Request one current preview frame from the active capture loop."""
    if constants.video_state.status != "running":
        raise HTTPException(status_code=409, detail="No active OCR video job is available to refresh")
    if not constants.preview_request_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A preview refresh is already in progress")

    try:
        constants.preview_completed.clear()
        constants.preview_requested.set()
        if not constants.preview_completed.wait(timeout=10):
            raise HTTPException(status_code=504, detail="Timed out waiting for a preview frame")
        if constants.video_state.status != "running":
            raise HTTPException(status_code=409, detail="The OCR video job stopped before preview refresh completed")
        with constants.preview_lock:
            metadata = constants.latest_preview_metadata
        if metadata is None:
            raise HTTPException(status_code=503, detail="The preview refresh did not produce a frame")
        return metadata
    finally:
        constants.preview_request_lock.release()

@app.get("/preview-frame")
def get_preview_frame(
    mode: Literal["raw", "threshold"] = "raw",
    _: None = Depends(require_admin_session),
) -> Response:
    """Return the latest authenticated raw or threshold preview JPEG."""
    if constants.video_state.status not in {"running", "stopping"}:
        raise HTTPException(status_code=409, detail="No active OCR video job is available")
    with constants.preview_lock:
        preview_jpeg = constants.latest_preview_jpeg
        preview_frame = constants.latest_preview_frame.copy() if constants.latest_preview_frame is not None else None
        metadata = constants.latest_preview_metadata
    if preview_jpeg is None or preview_frame is None or metadata is None:
        raise HTTPException(status_code=409, detail="No preview frame is available")
    if mode == "threshold":
        preview_jpeg = ocr.encode_preview_jpeg(ocr.threshold_preview_frame(preview_frame, ocr.current_calibration()))
    return Response(
        content=preview_jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Capture-Id": str(metadata.capture_id),
            "X-Frame-Width": str(metadata.frame_width),
            "X-Frame-Height": str(metadata.frame_height),
        },
    )

@app.get("/game-state-events")
async def game_state_events(request: Request) -> StreamingResponse:
    """Stream the latest and future game-state events to public viewers."""
    subscriber: Queue[str] = Queue(maxsize=1)

    with constants.event_lock:
        constants.event_subscribers.add(subscriber)
        if constants.latest_event:
            initial_event = constants.latest_event.model_dump_json(exclude_none=True)
        else:
            initial_event = GameStateEvent(
                id=0,
                timestamp=datetime.now(timezone.utc),
                status=cast(VideoJobStatus, constants.video_state.status),
                message=constants.video_state.message,
                game_state=GameState(score_state=ScoreState(), time_left=GameTimeState()),
            ).model_dump_json(exclude_none=True)

    async def event_stream():
        """Yield queued events and periodic SSE keepalive messages."""
        last_ping = time.monotonic()
        try:
            if initial_event:
                yield ocr.format_sse_event(initial_event)

            while True:
                try:
                    event_data = subscriber.get_nowait()
                    yield ocr.format_sse_event(event_data)
                    last_ping = time.monotonic()
                except Empty:
                    if time.monotonic() - last_ping >= 15.0:
                        yield ": keepalive\n\n"
                        last_ping = time.monotonic()
                    await asyncio.sleep(0.25)
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            with constants.event_lock:
                constants.event_subscribers.discard(subscriber)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/game-state")
def get_game_state() -> GameState:
    """Return the latest public game state or an empty initial state."""
    with constants.event_lock:
        game_state = constants.latest_event.game_state if constants.latest_event else None
    return game_state or GameState(score_state=ScoreState(), time_left=GameTimeState())


@app.post("/admin/session")
def create_admin_session(credentials: AdminPasswordRequest, response: Response) -> AdminSessionState:
    """Authenticate username and password and issue an admin session cookie."""
    if not SESSION_SECRET:
        raise HTTPException(status_code=503, detail="Admin authentication is not configured")
    if not is_valid_admin_credentials(credentials.username, credentials.password):
        raise HTTPException(status_code=403, detail="Invalid admin credentials")
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=create_admin_session_token(),
        max_age=SESSION_LIFETIME_SECONDS,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
    )
    return AdminSessionState(authenticated=True)


@app.get("/admin/session")
async def get_admin_session(
    request: Request,
    username: str | None = Query(default=None),
    password: str | None = Query(default=None),
    session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> AdminSessionState:
    """Report whether the request includes a valid admin session or credentials."""
    try:
        await require_admin_session(
            request=request,
            username=username,
            password=password,
            session_token=session_token,
        )
    except HTTPException:
        return AdminSessionState(authenticated=False)
    return AdminSessionState(authenticated=True)


@app.delete("/admin/session")
def delete_admin_session(response: Response) -> AdminSessionState:
    """Delete the caller's admin session cookie."""
    response.delete_cookie(SESSION_COOKIE_NAME, httponly=True, samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)
    return AdminSessionState(authenticated=False)

@app.post("/analyze-image-frame")
async def analyze_image_frame(
    file: UploadFile = File(...),
    _: None = Depends(require_admin_session),
) -> GameState | None:
    """Run authenticated OCR against an uploaded still image."""
    file_bytes: bytes = await file.read()
    nparr: NDArray[np.uint8] = np.frombuffer(file_bytes, np.uint8)
    original_img: np.ndarray | None = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original_img is not None:
        blue_coordinates, orange_coordinates, time_coordinates = ocr.get_coordinates(original_img)
        blue_metrics, orange_metrics, time_left = ocr.get_score(original_img, blue_coordinates, orange_coordinates, time_coordinates)
        return GameState(
            score_state=ScoreState(blue_score=blue_metrics.blue_score, orange_score=orange_metrics.orange_score),
            time_left=time_left,
        )
    return None


@app.post("/run-process-single-image")
def run_process_single_image(path_to_image: str, _: None = Depends(require_admin_session)) -> GameState | None:
    """Run authenticated OCR against a local static image file."""
    print(f"\n--- Testing Static Image: {path_to_image} ---")
    img = cv2.imread(path_to_image)
    if img is not None:
        blue_coordinates, orange_coordinates, time_coordinates = ocr.get_coordinates(img)
        blue_score_result, orange_score_result, time_left_result = ocr.get_score(img, blue_coordinates, orange_coordinates, time_coordinates)
        return GameState(
            score_state=ScoreState(blue_score=blue_score_result.blue_score, orange_score=orange_score_result.orange_score),
            time_left=time_left_result,
        )
    return None
class LiveStreamCapture:
    """Demux frames in a background thread with zero buffer lag, decoding on demand."""

    def __init__(self, source: str | int) -> None:
        self.cap = cv2.VideoCapture(source)
        self.is_file: bool = isinstance(source, str) and os.path.isfile(source)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_delay: float = 1.0 / fps if (self.is_file and fps and fps > 0) else 0.0
        self.running: bool = True
        self.stopped: bool = False
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self._decode_requested = threading.Event()
        self._decode_completed = threading.Event()
        self._retrieved_frame: np.ndarray | None = None
        self._retrieve_success: bool = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        if self.cap.isOpened():
            self.thread.start()

    def isOpened(self) -> bool:
        return self.cap.isOpened() and not self.stopped

    def get(self, prop: int) -> float:
        return self.cap.get(prop)

    def _reader(self) -> None:
        while self.running and self.cap.isOpened():
            grabbed = self.cap.grab()
            if not grabbed:
                time.sleep(0.01)
                continue
            self.new_frame_event.set()
            if self._decode_requested.is_set():
                ret, frame = self.cap.retrieve()
                with self.lock:
                    self._retrieve_success = ret
                    self._retrieved_frame = frame
                self._decode_requested.clear()
                self._decode_completed.set()
            if self.frame_delay > 0:
                time.sleep(self.frame_delay)
        with self.lock:
            self.stopped = True
            self._decode_requested.clear()
            self._decode_completed.set()

    def read_latest(self, timeout: float = 2.0) -> tuple[bool, np.ndarray | None]:
        """Request and retrieve the freshest frame on demand."""
        if not self.isOpened():
            return False, None
        self._decode_completed.clear()
        self._decode_requested.set()
        if not self._decode_completed.wait(timeout=timeout):
            self._decode_requested.clear()
            return False, None
        with self.lock:
            if not self._retrieve_success or self._retrieved_frame is None:
                return False, None
            return True, self._retrieved_frame.copy()

    def release(self) -> None:
        self.running = False
        self.stopped = True
        self._decode_requested.clear()
        self._decode_completed.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.cap.release()


def run_local_video(path_to_video: str | None, seconds_interval: float = 3.0, realtime: bool = True) -> None:
    """Capture frames, service refresh requests, and publish stabilized OCR state."""
    print(f"\n--- Testing Video File: {path_to_video} ---")
    live_cap = LiveStreamCapture(path_to_video if path_to_video else 0)

    if not live_cap.isOpened():
        print("Could not open video.")
        return

    fps: float = live_cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    print(f"Capture opened at {fps:.2f} FPS; OCR runs every {seconds_interval:.1f} seconds.")

    # Wait up to 5 seconds for the first frame to arrive
    live_cap.new_frame_event.wait(timeout=5.0)

    sample_count: int = 0
    next_sample_at = time.monotonic()
    state_reducer = ocr.GameStateReducer()

    while live_cap.isOpened():
        if constants.stop_requested.is_set():
            print("Stop requested; ending OCR job.")
            break

        if constants.reducer_reset_requested.is_set():
            state_reducer = ocr.GameStateReducer()
            constants.reducer_reset_requested.clear()

        refresh_was_requested = constants.refresh_requested.is_set()
        preview_was_requested = constants.preview_requested.is_set()
        now = time.monotonic()
        needs_sample = refresh_was_requested or (now >= next_sample_at)

        if not needs_sample and not preview_was_requested:
            if constants.stop_requested.is_set():
                break
            time.sleep(0.05)
            continue

        ret, frame = live_cap.read_latest()
        if not ret or frame is None:
            if constants.stop_requested.is_set():
                break
            time.sleep(0.05)
            continue

        if preview_was_requested:
            ocr.cache_preview_frame(frame)
            constants.preview_requested.clear()
            constants.preview_completed.set()

        if needs_sample:
            ocr.cache_preview_frame(frame)
            sample_count += 1
            t0 = time.monotonic()
            print(f"Starting OCR sample {sample_count} at {time.strftime('%H:%M:%S')}.")
            try:
                blue_score_results, orange_score_results, detected_time = ocr.get_ocr_result(frame)
                latest_result = state_reducer.update(
                    blue_score_results,
                    orange_score_results,
                    ocr.time_to_seconds(ocr.parse_detected_time(detected_time)),
                )
                print(f"Blue score: {latest_result.score_state.blue_score}")
                print(f"Orange score: {latest_result.score_state.orange_score}")
                display_minutes = latest_result.time_left.time_minutes
                display_seconds = latest_result.time_left.time_seconds
                ot_prefix = "+" if latest_result.time_left.is_overtime else ""
                formatted_minutes = str(display_minutes) if isinstance(display_minutes, int) else "xx"
                formatted_seconds = f"{display_seconds:02d}" if isinstance(display_seconds, int) else "xx"
                print(f"Time left: {ot_prefix}{formatted_minutes}:{formatted_seconds}")
                if latest_result.is_game_over and latest_result.winner:
                    print("=" * 45)
                    print(f"🏆 GAME OVER: {latest_result.winner} Team Wins ({latest_result.score_state.blue_score} - {latest_result.score_state.orange_score})!")
                    print("=" * 45)
                print(f"OCR sample {sample_count} completed in {time.monotonic() - t0:.2f} seconds.")
                event_msg = f"Game Over - {latest_result.winner} Team Wins!" if latest_result.is_game_over and latest_result.winner else "OCR update received"
                publish_game_state_event("running", event_msg, latest_result)
            except Exception as exc:
                if constants.stop_requested.is_set():
                    break
                print(f"OCR sample skipped: {exc}")
            finally:
                if refresh_was_requested:
                    constants.refresh_requested.clear()
                    constants.refresh_completed.set()
            next_sample_at = t0 + seconds_interval

    live_cap.release()

def _run_local_video_job(path_to_video: str | None, seconds_interval: float, realtime: bool) -> None:
    """Manage OCR job lifecycle state around the capture processing loop."""
    try:
        constants.refresh_requested.clear()
        constants.refresh_completed.clear()
        constants.preview_requested.clear()
        constants.preview_completed.clear()
        constants.reducer_reset_requested.clear()
        with constants.preview_lock:
            constants.latest_preview_jpeg = None
            constants.latest_preview_metadata = None
        constants.video_state.status = "running"
        constants.video_state.path_to_video = path_to_video
        constants.video_state.seconds_interval = seconds_interval
        constants.video_state.realtime = realtime
        constants.video_state.run_count = (constants.video_state.run_count or 0) + 1
        constants.video_state.last_error = None
        constants.video_state.message = "OCR video processing started"
        publish_game_state_event("running", "OCR video processing started")
        run_local_video(path_to_video if path_to_video else None, seconds_interval=seconds_interval, realtime=realtime)
    except Exception as exc:
        constants.video_state.last_error = str(exc)
        constants.video_state.message = f"OCR processing failed: {exc}"
        print("OCR processing failed:")
        traceback.print_exc()
        publish_game_state_event("failed", f"OCR processing failed: {exc}")
    finally:
        with constants.preview_lock:
            constants.latest_preview_jpeg = None
            constants.latest_preview_metadata = None
        constants.refresh_completed.set()
        constants.preview_completed.set()
        constants.video_state.status = "idle"
        constants.video_state.message = "OCR video processing stopped"
        if constants.video_state.last_error is None:
            publish_game_state_event("idle", "OCR video processing stopped")
        constants.video_state_lock.release()


@app.post("/run-local-video")
async def trigger_run_local_video(
    background_tasks: BackgroundTasks,
    start_request: StreamStartRequest | None = None,
) -> VideoState:
    """Start one background OCR job for a submitted stream."""
    request = start_request or StreamStartRequest()
    url = (request.stream_url or "").strip() or (constants.active_stream_url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Please enter a stream URL or video path")

    if not constants.video_state_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An OCR video job is already running")

    try:
        path_to_video = ocr.resolve_stream_url(url)
    except Exception:
        constants.video_state_lock.release()
        raise

    stream_changed = constants.active_stream_url != url
    constants.active_stream_url = url
    constants.stop_requested.clear()
    constants.reducer_reset_requested.clear()
    if stream_changed:
        with constants.calibration_lock:
            constants.active_calibration = OCRCalibration(
                blue_score=NormalizedRectangle(x=0.415, y=0.0, width=0.025, height=0.085),
                time=NormalizedRectangle(x=0.460, y=0.0, width=0.080, height=0.085),
                orange_score=NormalizedRectangle(x=0.555, y=0.0, width=0.035, height=0.085),
            )

    with constants.event_lock:
        constants.latest_event = None
    constants.video_state.status = "queued"
    constants.video_state.path_to_video = path_to_video
    constants.video_state.seconds_interval = request.seconds_interval
    constants.video_state.realtime = request.realtime
    constants.video_state.message = "run_local_video has been scheduled"
    publish_game_state_event("queued", "OCR video processing has been scheduled")
    background_tasks.add_task(_run_local_video_job, path_to_video, request.seconds_interval, request.realtime)
    return constants.video_state
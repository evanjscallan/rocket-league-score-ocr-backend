from datetime import datetime
import math
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


VideoJobStatus = Literal["queued", "running", "completed", "failed", "idle", "stopping"]


class VideoState(BaseModel):
    """Report the lifecycle and configuration of an OCR video job."""

    status: VideoJobStatus = "idle"
    message: str = "OCR video status is idle"
    path_to_video: str | None = None
    seconds_interval: float | None = 3.0
    realtime: bool | None = True
    run_count: int = Field(default=0, exclude=True)
    last_error: str | None = Field(default=None, exclude=True)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class StreamUrlRequest(BaseModel):
    """Validate a submitted stream URL."""

    stream_url: str | None = Field(default=None, max_length=2048)


class StreamStartRequest(BaseModel):
    """Validate stream input and sampling options for a new OCR job."""

    stream_url: str | None = Field(default=None, max_length=2048)
    seconds_interval: float = Field(default=3.0, ge=0.25, le=60.0)
    realtime: bool = True


class AdminPasswordRequest(BaseModel):
    """Carry the admin username and password used to create an admin session."""

    username: str = Field(default="admin", max_length=256)
    password: str = Field(default="admin", min_length=1, max_length=1024)


class AdminSessionState(BaseModel):
    """Report whether the caller has an active admin session."""

    authenticated: bool


class NormalizedRectangle(BaseModel):
    """Define an OCR crop as fractions of its source frame."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def stays_inside_frame(self) -> "NormalizedRectangle":
        """Reject rectangles that extend beyond normalized frame bounds."""
        if not all(math.isfinite(value) for value in (self.x, self.y, self.width, self.height)):
            raise ValueError("Rectangle values must be finite")
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("Rectangle must stay within the normalized frame bounds")
        return self


class OCRCalibration(BaseModel):
    """Group the three normalized regions required for score and timer OCR."""

    model_config = ConfigDict(extra="ignore")

    blue_score: NormalizedRectangle
    time: NormalizedRectangle
    orange_score: NormalizedRectangle


class PreviewFrameMetadata(BaseModel):
    """Identify a cached preview and its original frame dimensions."""

    capture_id: int
    frame_width: int
    frame_height: int


class ScoreState(BaseModel):
    """Represent the detected score for both Rocket League teams."""

    orange_score: int | None | Literal["N/A"] = 0
    blue_score: int | None | Literal["N/A"] = 0


class GameTimeState(BaseModel):
    """Represent the detected remaining game-clock value and overtime status."""

    time_minutes: int | None | Literal["xx"] = None
    time_seconds: int | None | Literal["xx"] = None
    is_overtime: bool = False


class GameState(BaseModel):
    """Combine stabilized score, game-clock state, and winner detection."""

    score_state: ScoreState
    time_left: GameTimeState
    winner: Literal["Blue", "Orange"] | None = None
    is_game_over: bool = False


class GameStateEvent(BaseModel):
    """Represent a timestamped game-state lifecycle event for SSE clients."""

    id: int
    timestamp: datetime
    status: VideoJobStatus
    message: str
    game_state: GameState | None = None
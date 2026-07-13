from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr

from .models import MessageRole, VideoStatus


# ---- Auth ----
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: EmailStr
    username: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AuthResponse(BaseModel):
    user: UserOut
    token: Token


# ---- Videos ----
class DetectionResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    flower_map: int
    insect_tracks: int
    pollinator_visits: int
    non_pollinator_visits: int
    flower_map50: float
    insect_map50: float
    classifier_acc: float


class VisitOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    flower_id: int
    insect_class: str
    is_pollinator: bool
    dwell_sec: float
    first_seen_sec: float


class VideoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    original_name: str
    status: VideoStatus
    error: str | None
    duration_sec: float | None
    uploaded_at: datetime
    processed_at: datetime | None
    thumbnail_path: str | None
    result: DetectionResultOut | None = None


class VideoStatusOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    status: VideoStatus
    error: str | None


# ---- Stats ----
class StatsOverview(BaseModel):
    videos_processed: int
    total_visits: int
    pollinator_pct: float
    avg_visits_per_flower: float
    top_flower: int | None


class FlowerVisitBar(BaseModel):
    flower_id: int
    total: int
    pollinator: int
    non_pollinator: int


class StatsVisits(BaseModel):
    bars: list[FlowerVisitBar]
    total: int
    pollinator: int
    non_pollinator: int


class TimeseriesPoint(BaseModel):
    bucket: str
    visits: int


class StatsTimeseries(BaseModel):
    points: list[TimeseriesPoint]


# ---- Chat ----
class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    updated_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    role: MessageRole
    content: str
    created_at: datetime


class ConversationDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]


class MessageCreate(BaseModel):
    content: str
    provider: str | None = None   # "auto" | "gemini" | "huggingface" | "anthropic" | "mock"

"""Aggregation / filter logic for the Stats section."""

from collections import defaultdict
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import DetectionResult, Video, VideoStatus, Visit
from ..schemas import (
    FlowerVisitBar,
    StatsOverview,
    StatsTimeseries,
    StatsVisits,
    TimeseriesPoint,
)


def _user_visit_query(user_id: int):
    return (
        select(Visit)
        .join(Video, Visit.video_id == Video.id)
        .where(Video.user_id == user_id)
    )


def overview(db: Session, user_id: int) -> StatsOverview:
    processed_count = len(
        db.scalars(
            select(Video.id).where(
                Video.user_id == user_id, Video.status == VideoStatus.done
            )
        ).all()
    )

    visits = db.scalars(_user_visit_query(user_id)).all()
    total = len(visits)
    pollinator = sum(1 for v in visits if v.is_pollinator)
    pollinator_pct = round(100 * pollinator / total, 1) if total else 0.0

    per_flower: dict[int, int] = defaultdict(int)
    for v in visits:
        per_flower[v.flower_id] += 1
    avg_per_flower = round(total / len(per_flower), 2) if per_flower else 0.0
    top_flower = max(per_flower, key=per_flower.get) if per_flower else None

    return StatsOverview(
        videos_processed=processed_count,
        total_visits=total,
        pollinator_pct=pollinator_pct,
        avg_visits_per_flower=avg_per_flower,
        top_flower=top_flower,
    )


def visits(
    db: Session,
    user_id: int,
    video_id: int | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    pollinator: bool | None = None,
) -> StatsVisits:
    q = _user_visit_query(user_id)
    if video_id is not None:
        q = q.where(Visit.video_id == video_id)
    if pollinator is not None:
        q = q.where(Visit.is_pollinator == pollinator)
    if date_from is not None:
        q = q.where(Video.uploaded_at >= date_from)
    if date_to is not None:
        q = q.where(Video.uploaded_at <= date_to)

    rows = db.scalars(q).all()

    agg: dict[int, dict[str, int]] = defaultdict(
        lambda: {"pollinator": 0, "non_pollinator": 0}
    )
    for v in rows:
        key = "pollinator" if v.is_pollinator else "non_pollinator"
        agg[v.flower_id][key] += 1

    bars = [
        FlowerVisitBar(
            flower_id=fid,
            total=counts["pollinator"] + counts["non_pollinator"],
            pollinator=counts["pollinator"],
            non_pollinator=counts["non_pollinator"],
        )
        for fid, counts in sorted(agg.items())
    ]
    total_poll = sum(b.pollinator for b in bars)
    total_non = sum(b.non_pollinator for b in bars)
    return StatsVisits(
        bars=bars,
        total=total_poll + total_non,
        pollinator=total_poll,
        non_pollinator=total_non,
    )


def timeseries(db: Session, user_id: int, bucket: str = "day") -> StatsTimeseries:
    rows = db.scalars(
        _user_visit_query(user_id).join(
            DetectionResult,
            DetectionResult.video_id == Visit.video_id,
            isouter=True,
        )
    ).all()

    counts: dict[str, int] = defaultdict(int)
    for v in rows:
        day = v.video.uploaded_at.strftime("%Y-%m-%d")
        counts[day] += 1

    points = [
        TimeseriesPoint(bucket=b, visits=c) for b, c in sorted(counts.items())
    ]
    return StatsTimeseries(points=points)

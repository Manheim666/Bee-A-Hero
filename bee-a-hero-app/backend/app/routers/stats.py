from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..db import get_db
from ..models import User
from ..schemas import StatsOverview, StatsTimeseries, StatsVisits
from ..services import stats as stats_service

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/overview", response_model=StatsOverview)
def overview(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return stats_service.overview(db, user.id)


@router.get("/visits", response_model=StatsVisits)
def visits(
    video_id: int | None = None,
    from_: datetime | None = Query(None, alias="from"),
    to: datetime | None = None,
    pollinator: bool | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return stats_service.visits(
        db,
        user.id,
        video_id=video_id,
        date_from=from_,
        date_to=to,
        pollinator=pollinator,
    )


@router.get("/timeseries", response_model=StatsTimeseries)
def timeseries(
    bucket: str = "day",
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return stats_service.timeseries(db, user.id, bucket=bucket)

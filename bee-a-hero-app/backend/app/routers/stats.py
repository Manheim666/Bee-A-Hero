from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..db import get_db
from ..models import User
from ..schemas import StatsOverview, StatsTimeseries, StatsVisits
from ..services import stats as stats_service
from ..services import yield_model

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/crops")
def crops():
    """Crop menu for the UI — each crop has its own fruit-set formula (see yield_model)."""
    return {"crops": yield_model.crop_list(), "default": yield_model.DEFAULT_CROP}


@router.get("/yield")
def yield_estimate(
    crop: str = yield_model.DEFAULT_CROP,
    n_flowers: int = 1000,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Per-crop pollination -> fruit-set -> yield from the user's detected visits."""
    rows = db.scalars(stats_service._user_visit_query(user.id)).all()
    visits = [{"insect_class": r.insect_class, "dwell_sec": r.dwell_sec,
               "is_pollinator": r.is_pollinator} for r in rows]
    return yield_model.estimate(visits, crop=crop, n_flowers=n_flowers)


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

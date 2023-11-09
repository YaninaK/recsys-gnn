import logging
import pandas as pd
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ["train_valid_split"]


VALID_WEEKS = 3


def train_valid_split(data: pd.DataFrame, valid_weeks: Optional[int] = None):
    if valid_weeks is None:
        valid_weeks = VALID_WEEKS

    t = data["week_no"].max() - valid_weeks
    train = data[data["week_no"] <= t]
    valid = data[data["week_no"] > t]

    return train, valid

import logging
import pandas as pd
import numpy as np
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ["prefilter_items"]


LOWER_PRICE_THRESHOLD = 1
UPPER_PRICE_THRESHOLD = 30
TAKE_N_POPULAR = 5000


def prefilter_items(
    X: pd.DataFrame,
    lower_price_threshold: Optional[int] = None,
    upper_price_threshold: Optional[int] = None,
    take_n_popular: Optional[int] = None,
) -> pd.DataFrame:
    """
    1. Selects data starting from the 20th week.
    2. Excludes items that have not been sold for the last 12 months.
    3. Excludes top popular items.
    4. Excludes top unpopular items.
    5. Excludes cheap items.
    6. Excluding expensive items.
    7. Selects N popular items.
    """

    if lower_price_threshold is None:
        lower_price_threshold = LOWER_PRICE_THRESHOLD
    if upper_price_threshold is None:
        upper_price_threshold = UPPER_PRICE_THRESHOLD
    if take_n_popular is None:
        take_n_popular = TAKE_N_POPULAR

    data = X.copy()

    logging.info("Selecting data starting from the 20th week...")

    data = data[data["week_no"] >= 20]

    logging.info("Excluding items that have not been sold for the last 12 months...")

    t = max(data["day"]) - 365
    not_sold_in_12_month = list(
        set(data[data["day"] < t]["item_id"].unique())
        - set(data[data["day"] >= t]["item_id"].unique())
    )
    data = data[~data["item_id"].isin(not_sold_in_12_month)]

    logging.info("Excluding top popular items ...")

    n_users = data["user_id"].nunique()
    popularity = data.groupby("item_id", as_index=False).agg(
        share_unique_users=("user_id", lambda x: x.nunique() / n_users)
    )
    top_popular = popularity[popularity["share_unique_users"] > 0.2].item_id.tolist()
    data = data[~data["item_id"].isin(top_popular)]

    logging.info("Excluding top unpopular items ...")

    top_unpopular = popularity[popularity["share_unique_users"] < 0.01].item_id.tolist()
    data = data[~data["item_id"].isin(top_unpopular)]

    logging.info("Excluding cheap items ...")

    data.loc[:, "price"] = (
        data["sales_value"] - data["retail_disc"] - data["coupon_match_disc"]
    ) / np.maximum(data["quantity"], 1)
    cheap_products = (
        data.loc[data["price"] < lower_price_threshold, "item_id"].unique().tolist()
    )
    data = data[~data["item_id"].isin(cheap_products)]

    logging.info("Excluding expensive items ...")

    expensive_products = (
        data.loc[data["price"] > upper_price_threshold, "item_id"].unique().tolist()
    )
    data = data[~data["item_id"].isin(expensive_products)]

    logging.info(f"Taking {take_n_popular} popular items ...")

    popular = (
        data.groupby("item_id", as_index=False)
        .agg(n_users=("user_id", "count"))
        .sort_values("n_users", ascending=False)
        .head(take_n_popular)["item_id"]
        .tolist()
    )
    data.loc[~data["item_id"].isin(popular), "item_id"] = 999999

    return data

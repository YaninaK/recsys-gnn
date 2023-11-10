import logging
import pandas as pd
import numpy as np
from typing import Optional
import re


logger = logging.getLogger(__name__)

__all__ = ["prefilter_items"]

RENAME_LIST = [
    [r"OZ", r" OZ "],
    [r"0Z", r" OZ "],
    [r"OZS", r" OZ "],
    [r"OUNCE", r"OZ"],
    [r"LBS", r"LB"],
    [r"LB", r" LB "],
    [r"CT", r" CT "],
    [r"INCH", r" IN "],
    [r"IN", r" IN "],
    [r"SQ", r" SQ "],
    [r"FT", r" FT "],
    [r"PK", r" PK "],
    [r"LOADS", r"LOAD"],
    [r"LDS", r" LOAD "],
    [r"LD", r" LOAD "],
    [r"SHEETS", r"SHEET"],
    [r"USES", r"USE"],
    [r"STEMS", r"STEM"],
    [r"STEM", r" STEM "],
    [r"LITER", r"LTR"],
    [r"LTR", r" LTR "],
    [r"LT", r" LTR "],
    [r"ML", r" ML "],
    [r"QUART", r"QT"],
    [r"QT", r" QT "],
    [r"GL", r"GAL"],
    [r"GALLON", r"GAL"],
    [r"LRG/XLRG", r"X-LG"],
    [r"X-LARGE", r"X-LG"],
    [r"2XL", r"X-LG"],
    [r"XL", r"X-LG"],
    [r"CM", r" CM "],
    [r"MM", r" MM "],
    [r"YD", r" YD "],
    [r"DOSES", r" DOSES "],
]


def preprocess_curr_size_of_product(
    item_features: pd.DataFrame,
    rename_list: Optional[list] = None,
) -> pd.DataFrame:

    if rename_list is None:
        rename_list = RENAME_LIST

    feature = "curr_size_of_product"
    df = pd.DataFrame(item_features[feature].unique(), columns=[feature])

    new_feature = "adj_" + feature
    df[new_feature] = df[feature].copy()
    for i, j in rename_list:
        df[new_feature] = df[new_feature].apply(lambda text: re.sub(i, j, text))

    return df

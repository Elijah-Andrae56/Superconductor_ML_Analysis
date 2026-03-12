
from __future__ import annotations

import pandas as pd


def display_metrics(metrics: dict, title: str = "Metrics") -> pd.DataFrame:
    """Return a dataframe for cleaner notebook display."""
    return pd.DataFrame(metrics, index=[title]).T.rename(columns={title: "value"})

import numpy as np
import pandas as pd

from tests_lib.analysis_utils import AnalysisUtils


def test_get_hourly_average_value_ignores_non_numeric_columns():
    df = pd.DataFrame(
        {
            "TimeStamp": [
                "2024-01-01_00:00:00",
                "2024-01-01_00:01:00",
                "2024-01-01_01:00:00",
            ],
            "value": [1.0, 2.0, 3.0],
            "flag": ["x", "y", "z"],
        }
    )

    utils = AnalysisUtils()
    prepared = utils.get_data_for_day_hour(df, device="power_card")

    averages, stds = utils.get_hourly_average_value(
        data_frame=prepared,
        column="value",
        min_scale=None,
        unique_days=["2024-01-01"],
    )

    assert len(averages) == 2
    assert len(stds) == 2
    assert np.isclose(averages[0], 1.5)
    assert np.isclose(averages[1], 3.0)

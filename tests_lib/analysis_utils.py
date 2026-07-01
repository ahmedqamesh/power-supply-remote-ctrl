########################################################
"""
    Author: Ahmed Qamesh
    email: ahmed.qamesh@cern.ch
    Date: 29.01.2022
    Optimised: July 2026
"""
########################################################
from __future__ import division
import os
import csv
import yaml
import socket
import ipaddress
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
try:
    from .logger_main import Logger
    log_format = '%(log_color)s[%(levelname)s]  - %(name)s -%(message)s'
    log_call = Logger(log_format=log_format, name="Analysis",
                      console_loglevel=logging.INFO, logger_file=False)
    logger = log_call.setup_main_logger()
except Exception:
    logging.basicConfig(format="[%(levelname)s] %(name)s - %(message)s",
                        level=logging.INFO)
    logger = logging.getLogger("AnalysisUtils")

# Timestamp formats — defined once, used everywhere
_TS_FMT_POWER = "%Y-%m-%d_%H:%M:%S"
_TS_FMT_OTHER = "%Y-%m-%d %H:%M:%S.%f"


class AnalysisUtils(object):

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def open_yaml_file(self, directory=None, file=None):
        with open(os.path.join(directory, file), "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def dump_yaml_file(self, directory=None, file=None, loaded=None):
        with open(os.path.join(directory, file), "w") as f:
            yaml.dump(loaded, f, sort_keys=False)

    def open_csv_file(self, outputname=None, directory=None):
        os.makedirs(directory, exist_ok=True)
        return open(os.path.join(directory, outputname) + ".csv", "w+")

    def build_data_base(self, fieldnames=("A", "B"), outputname=False, directory=False):
        out_file = self.open_csv_file(outputname=outputname, directory=directory)
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        return csv.writer(out_file), out_file

    def save_to_csv(self, data=None, outname=None, directory=None):
        os.makedirs(directory, exist_ok=True)
        pd.DataFrame(data).to_csv(os.path.join(directory, outname), index=True)

    def read_csv_file(self, file=None):
        return pd.read_csv(file, encoding="utf-8").fillna(0)

    # ------------------------------------------------------------------
    # Sentinel / last-row check
    # ------------------------------------------------------------------

    def check_last_row(self, data_frame=None, column="status"):
        """
        Drop the 'End of Test' sentinel row if present.

        FIX (original): used skipfooter=2 in combine_csv_files, which
        silently ate the last real data row as well.
        """
        last_val = str(data_frame.iloc[-1][column])
        if "End of Test" in last_val:
            logger.info("Complete test file detected.")
            return data_frame.iloc[:-1].reset_index(drop=True)
        logger.warning("Incomplete test file — no 'End of Test' sentinel found.")
        return data_frame

    # ------------------------------------------------------------------
    # Timestamp parsing  (single helper — no more double pd.to_datetime)
    # ------------------------------------------------------------------

    def _parse_timestamps(self, series: pd.Series, device: str = "power_card") -> pd.Series:
        """Parse a raw timestamp column into a DatetimeSeries (once)."""
        fmt = _TS_FMT_POWER if device == "power_card" else _TS_FMT_OTHER
        return pd.to_datetime(series, format=fmt, errors="coerce")

    def get_data_for_day_hour(self, data_frame=None, device="power_card"):
        """
        Enrich *data_frame* with 'day', 'hour', 'minutes' columns.

        FIX (original): called pd.to_datetime twice on the same column.
        """
        df = data_frame.copy()
        ts = self._parse_timestamps(df["TimeStamp"], device=device)
        df["TimeStamp"] = ts
        df["day"]       = ts.dt.date
        df["hour"]      = ts.dt.hour
        df["minutes"]   = ts.dt.minute
        return df

    # ------------------------------------------------------------------
    # getDay
    # ------------------------------------------------------------------

    def getDay(self, TimeStamps):
        """
        Return (first_timestamp_string, list_of_unique_date_strings).

        FIX (original): used a character-by-character while-loop just to
        copy TimeStamps[0], and built a 'days' list it never returned.
        """
        day = TimeStamps.iloc[0] if hasattr(TimeStamps, "iloc") else TimeStamps[0]
        ts_parsed   = pd.to_datetime(TimeStamps, format=_TS_FMT_POWER, errors="coerce")
        date_strings = ts_parsed.dt.strftime("%Y-%m-%d").dropna()
        seen = {}
        unique_days = [seen.setdefault(d, d) for d in date_strings if d not in seen]
        return day, unique_days

    # ------------------------------------------------------------------
    # getHours
    # ------------------------------------------------------------------

    def getHours(self, TimeStamps, min_scale=None, device=None):
        """
        Return (values_array, unique_hours, unique_minutes).

        FIX (original): used fragile string slicing ([11:13], [14:16]) that
        breaks on any format variation; had a duplicated else-branch that ran
        even inside the power_card if-block, producing wrong results.
        """
        ts = pd.to_datetime(TimeStamps, format=_TS_FMT_POWER, errors="coerce")

        if min_scale == "min_scale":
            values        = ts.dt.minute
            unique_values = sorted(values.dropna().unique().astype(int).tolist())
            return values.tolist(), [], unique_values
        else:
            values        = ts.dt.hour
            unique_values = sorted(values.dropna().unique().astype(int).tolist())
            return values.tolist(), unique_values, []

    # ------------------------------------------------------------------
    # Averaging — single correct implementation
    # ------------------------------------------------------------------

    def get_hourly_average_value(self, data_frame=None, column=None,
                                  min_scale=None, unique_days=None,
                                  device="power_card"):
        """
        Return (mean_array, std_array) grouped by hour or minute.

        FIX (original getHourlyAverageValue): pre-allocated dataSum with
        len(unique_hours) slots but indexed by the raw hour integer (e.g. 20),
        causing IndexError for any non-zero-based hour; bare try/except
        silently swallowed those errors.
        """
        df = self.get_data_for_day_hour(data_frame=data_frame, device=device)

        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        df = df.copy()
        df[column] = pd.to_numeric(df[column], errors="coerce")

        group_key = ["hour", "minutes"] if min_scale == "min_scale" else "hour"
        days       = unique_days if unique_days is not None else df["day"].unique()

        means, stds = [], []
        for target_day in days:
            day_data = df[df["day"] == pd.to_datetime(target_day).date()]
            if day_data.empty:
                continue
            grp = day_data.groupby(group_key, dropna=False)[column]
            means.extend(grp.mean().tolist())
            stds.extend(grp.std().tolist())

        return np.asarray(means, dtype=float), np.asarray(stds, dtype=float)

    # Deprecated alias — keeps old call-sites alive
    def getHourlyAverageValue(self, hours=None, data=None, min_scale=None,
                               unique_hours=None, unique_days=None, device="power_card"):
        """
        Deprecated — delegates to get_hourly_average_value.

        FIX (original): indexed pre-allocated list by raw hour int → IndexError;
        bare try/except hid the crash and silently dropped data.
        """
        logger.warning("getHourlyAverageValue() is deprecated; "
                       "use get_hourly_average_value() instead.")
        df = pd.DataFrame({
            "TimeStamp": pd.NaT,
            "hour":      pd.to_numeric(pd.Series(hours),  errors="coerce"),
            "_value":    pd.to_numeric(pd.Series(data),   errors="coerce"),
            "day":       pd.Timestamp("1970-01-01").date(),
            "minutes":   0,
        })
        group_key = "minutes" if min_scale == "min_scale" else "hour"
        grp   = df.groupby(group_key, dropna=False)["_value"]
        return grp.mean().to_numpy(dtype=float), grp.std().to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Consecutive repeat counter
    # ------------------------------------------------------------------

    def count_consecutive_repeats(self, timestamps=None):
        """
        Count runs of identical values.

        Cleaned up: replaced manual index arithmetic with a clear while-window.
        """
        if not timestamps:
            return [], [], []

        consecutive_repeats, repeats, new_array = [], [], []
        run_idx = 0
        i, n = 0, len(timestamps)

        while i < n:
            j = i + 1
            while j < n and timestamps[j] == timestamps[i]:
                j += 1
            count = j - i
            consecutive_repeats.append(run_idx)
            repeats.append(count)
            new_array.extend([run_idx] * count)
            run_idx += 1
            i = j

        return consecutive_repeats, repeats, new_array

    # ------------------------------------------------------------------
    # CSV combiner
    # ------------------------------------------------------------------

    def combine_csv_files(self, *input_files):
        """
        Concatenate CSVs, stripping the 'End of Test' sentinel from each.

        FIX (original): used skipfooter=2 which dropped the sentinel AND
        the last real data row.
        """
        frames = []
        for path in input_files:
            df   = pd.read_csv(path)
            mask = df.apply(
                lambda row: row.astype(str).str.contains("End of Test").any(),
                axis=1,
            )
            frames.append(df[~mask])

        combined  = pd.concat(frames, ignore_index=True)
        out_path  = os.path.join(os.path.dirname(input_files[0]), "combined.csv")
        combined.to_csv(out_path, index=False)
        return combined

    # ------------------------------------------------------------------
    # Network helpers  (unchanged)
    # ------------------------------------------------------------------

    def get_ip_device_address(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip

    def get_ip_from_subnet(self, ip_subnet):
        return [str(ip) for ip in ipaddress.ip_network(ip_subnet)]

    # ------------------------------------------------------------------
    # YAML index helpers  (unchanged)
    # ------------------------------------------------------------------

    def get_subindex_description_yaml(self, dictionary=None, index=None, subindex=None):
        return dictionary[index]["subindex_items"][subindex]

    def get_info_yaml(self, dictionary=None, index=None, subindex="description_items"):
        return dictionary[index][subindex]

    def get_subindex_yaml(self, dictionary=None, index=None, subindex="subindex_items"):
        return dictionary[index][subindex].keys()
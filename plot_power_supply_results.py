########################################################
"""
    Author: Ahmed Qamesh
    email: ahmed.qamesh@cern.ch
    Date: 29.08.2023
    Optimised: July 2026
"""
########################################################

import os
import re
import glob
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
import click

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Project imports — Logger and AnalysisUtils from tests_lib.
# A lightweight fallback is provided so the script can also run standalone
# (e.g. for quick tests without the full tests_lib environment).
# ---------------------------------------------------------------------------
try:
    from tests_lib.analysis_utils import AnalysisUtils
    from tests_lib.logger_main    import Logger
    from tests_lib.plot_style     import *  # noqa: F401,F403  (project-wide style)

    log_format = '%(log_color)s[%(levelname)s]  - %(name)s -%(message)s'
    log_call   = Logger(log_format=log_format, name="Plotting",
                        console_loglevel=logging.INFO, logger_file=False)
    logger     = log_call.setup_main_logger()
except Exception:
    logging.basicConfig(format="[%(levelname)s] %(name)s - %(message)s",
                        level=logging.INFO)
    logger = logging.getLogger("PowerSupplyPlotter")

    # Inline fallback AnalysisUtils (used only when tests_lib is absent)
    class AnalysisUtils:                          # type: ignore[no-redef]
        _TS_FMT = "%Y-%m-%d_%H:%M:%S"

        def check_last_row(self, data_frame, column="status"):
            last = str(data_frame.iloc[-1][column])
            if "End of Test" in last:
                return data_frame.iloc[:-1].reset_index(drop=True)
            return data_frame

        def getDay(self, ts):
            day = ts.iloc[0] if hasattr(ts, "iloc") else ts[0]
            parsed = pd.to_datetime(ts, format=self._TS_FMT, errors="coerce")
            dates  = parsed.dt.strftime("%Y-%m-%d").dropna()
            seen   = {}
            return day, [seen.setdefault(d, d) for d in dates if d not in seen]

        def get_hourly_average_value(self, data_frame, column, min_scale=None,
                                     unique_days=None, device="power_card"):
            df  = data_frame.copy()
            ts  = pd.to_datetime(df["TimeStamp"], format=self._TS_FMT, errors="coerce")
            df["day"]     = ts.dt.date
            df["hour"]    = ts.dt.hour
            df["minutes"] = ts.dt.minute
            df[column]    = pd.to_numeric(df[column], errors="coerce")
            gk  = ["hour", "minutes"] if min_scale == "min_scale" else "hour"
            days = unique_days or df["day"].unique()
            m, s = [], []
            for d in days:
                sub = df[df["day"] == pd.to_datetime(d).date()]
                if sub.empty: continue
                grp = sub.groupby(gk, dropna=False)[column]
                m.extend(grp.mean()); s.extend(grp.std())
            return np.asarray(m, float), np.asarray(s, float)

os.environ["NUMEXPR_MAX_THREADS"] = "4"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Timestamp format (must match what AnalysisUtils expects)
_TS_FMT = "%Y-%m-%d_%H:%M:%S"

# ---------------------------------------------------------------------------
# Time-scale auto-detection
# ---------------------------------------------------------------------------

def detect_time_scale(data_frame: pd.DataFrame, override: str = None):
    """
    Determine the appropriate time axis from the actual data span.

    Rules
    -----
    span <  5 min   →  raw elapsed seconds   (sec_scale)
    span <  1 hour  →  minutes from start    (min_scale)
    span >= 1 hour  →  hours from start      (hour_scale)

    *override* lets the user force a specific scale via --scale.

    Returns (scale_key, x_label, x_values_array, min_scale_str_for_averaging)
    where min_scale_str_for_averaging matches the AnalysisUtils convention
    ("min_scale" | "hour_scale" | None).
    """
    ts = pd.to_datetime(data_frame["TimeStamp"], format=_TS_FMT, errors="coerce").dropna()

    if ts.empty:
        # Fall back to elapsed_time column
        elapsed = pd.to_numeric(data_frame.get("elabsed_time", pd.Series()), errors="coerce").dropna()
        x = elapsed.to_numpy() if not elapsed.empty else np.arange(len(data_frame))
        return "sec_scale", "Time [Sec]", x, None

    t0      = ts.min()
    span_s  = (ts.max() - t0).total_seconds()
    sec_arr = (ts - t0).dt.total_seconds().to_numpy()

    if override:
        if override == "min_scale":
            return "min_scale",  "Time [Min]",  sec_arr / 60.0,   "min_scale"
        if override == "hour_scale":
            return "hour_scale", "Time [Hour]", sec_arr / 3600.0, "hour_scale"
        return "sec_scale", "Time [Sec]", sec_arr, None

    if span_s < 300:
        logger.info(f"  Span {span_s:.0f} s → seconds axis")
        return "sec_scale", "Time [Sec]", sec_arr, None
    elif span_s < 3600:
        logger.info(f"  Span {span_s/60:.1f} min → minutes axis")
        return "min_scale", "Time [Min]", sec_arr / 60.0, "min_scale"
    else:
        logger.info(f"  Span {span_s/3600:.2f} h → hours axis")
        return "hour_scale", "Time [Hour]", sec_arr / 3600.0, "hour_scale"


def _use_raw(data_frame: pd.DataFrame) -> bool:
    """True when the data span is too short to bin into hour/minute averages."""
    ts   = pd.to_datetime(data_frame["TimeStamp"], format=_TS_FMT, errors="coerce").dropna()
    return ts.empty or (ts.max() - ts.min()).total_seconds() < 3600


# ---------------------------------------------------------------------------
# Data analysis helpers
# ---------------------------------------------------------------------------

def calculate_efficiency_errors(Uout, Iout, Uin, Iin, eUout, eIout, eIin):
    """Compute efficiency, power, and their propagated errors."""
    efficiency = (Uout * Iout * 100) / (Uin * Iin)
    efficiency[efficiency.isnull()] = 0

    d_dUout = (Iout * 100) / (Uin * Iin)
    d_dIout = (Uout * 100) / (Uin * Iin)
    d_dUin  = (-Uout * Iout * 100) / (Uin ** 2 * Iin)
    d_dIin  = (-Uout * Iout * 100) / (Uin * Iin ** 2)

    delta_epsilon = np.sqrt(
        (d_dUout * eUout) ** 2
        + (d_dIout * eIout) ** 2
        + (d_dUin  * np.std(Uin)) ** 2
        + (d_dIin  * eIin) ** 2
    )
    power       = Uout * Iout * 0.001
    delta_power = np.sqrt((Iout * eUout) ** 2 + (Uout * eIout) ** 2) * 0.001
    return efficiency, delta_epsilon, power, delta_power


# Module-level singleton — AnalysisUtils is stateless, no need to reinstantiate
_au = AnalysisUtils()


def load_data(data_file: str):
    """Load CSV, skip the units row, drop 'End of Test' sentinel."""
    df      = pd.read_csv(data_file, skiprows=[1])
    headers = df.columns.tolist()
    df      = _au.check_last_row(data_frame=df, column=headers[-1])
    return df, headers


def extract_data_info(data_file: str, min_scale_key: str = None):
    """
    Load a CSV and compute per-column averages (binned or raw).

    FIX (original): called getHours() and discarded the return value.
    FIX: x-axis derivation now delegated entirely to detect_time_scale.
    """
    data_frame, df_headers = load_data(data_file)
    _, unique_days          = _au.getDay(data_frame.TimeStamp)

    n_cols    = len(df_headers)
    avg_vals  = [None] * n_cols
    std_vals  = [None] * n_cols
    new_hdrs  = []
    use_raw   = _use_raw(data_frame)

    for pos, col in enumerate(df_headers):
        if 0 < pos < n_cols:
            if use_raw:
                v = pd.to_numeric(data_frame[col], errors="coerce").dropna().to_numpy()
                avg_vals[pos - 1] = v
                std_vals[pos - 1] = np.zeros_like(v, dtype=float)
            else:
                avg_vals[pos - 1], std_vals[pos - 1] = _au.get_hourly_average_value(
                    data_frame=data_frame,
                    column=col,
                    min_scale=min_scale_key,
                    unique_days=unique_days,
                )
            new_hdrs.append(col)

    return data_frame, new_hdrs, avg_vals, std_vals


def _to_array(value, factor: float = 1.0) -> np.ndarray:
    """Coerce an avg_vals entry (ndarray or scalar list) to a flat float array."""
    return np.asarray(value, dtype=float) * factor


def _sanitize_filename(text: str) -> str:
    """Remove characters that are unsafe in filenames (LaTeX, slashes, etc.)."""
    return re.sub(r'[\\/*?:"<>|${}^]', "", text).strip().replace(" ", "_")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_power_supply_parameters(
    csv_files: list,
    legends:   list,
    pdf_pages: PdfPages,
    scale_override: str = None
):
    """
    Plot supply voltage and current for every CSV, overlaid on shared axes.
    """
    logger.info(f"Plotting {len(csv_files)} file(s)")

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.4)
    ax1.set_ylabel("$V_{Supply}$ [V]")
    ax2.set_ylabel("$I_{Supply}$ [mA]")

    ax1.set_title("Supply Voltage during Proton Irradiation")
    ax2.set_title("Supply Current during Proton Irradiation")

    all_voltages: list = []
    all_currents: list = []

    for data_file, legend_title in zip(csv_files, legends):
        logger.info(f"  Processing: {data_file}")
        stem = os.path.splitext(os.path.basename(data_file))[0]

        # ---- load & detect time axis --------------------------------
        try:
            data_frame, new_hdrs, avg_vals, std_vals = extract_data_info(
                data_file, min_scale_key=scale_override
            )
        except Exception as exc:
            logger.error(f"  Cannot process {data_file}: {exc}")
            continue

        _, time_label, x_values, _ = detect_time_scale(data_frame, override=scale_override)

        # ---- extract V and I series ---------------------------------
        # Column layout: TimeStamp | elapsed | Usin1 | eUsin1 | … | Isin1 | eIsin1 | …
        voltage_arr = _to_array(avg_vals[1])           # col index 2 → avg index 1
        current_arr = _to_array(avg_vals[5], 1000.0)   # col index 6 → avg index 5, A→mA

        # Align x to whichever series is shorter (raw values may differ after dropna)
        n = min(len(x_values), len(voltage_arr), len(current_arr))
        xv = x_values[:n]
        va = voltage_arr[:n]
        ia = current_arr[:n]

        if ia.size > 0 and np.any(ia != 0):
            delta_i = (np.max(ia) - np.min(ia)) / np.max(ia) * 100
            logger.info(f"  ΔI = {delta_i:.2f} %")
        # ---- accumulate for overlay axes ----------------------------
        ax1.plot(xv, va, label=legend_title, marker="o")
        ax2.plot(xv, ia, label=legend_title, marker="o")
        ax1.set_xlabel(time_label)
        ax2.set_xlabel(time_label)
        ax1.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        all_voltages.append(va[np.isfinite(va)])
        all_currents.append(ia[np.isfinite(ia)])

    # ---- Y-limits set ONCE after all series are drawn ---------------
    if all_voltages:
        flat_v = np.concatenate(all_voltages)
        if flat_v.size:
            margin_v = max((flat_v.max() - flat_v.min()) * 0.2, 0.01)
            ax1.set_ylim(flat_v.min() - margin_v, flat_v.max() + margin_v)
    if all_currents:
        flat_i = np.concatenate(all_currents)
        if flat_i.size:
            margin_i = max((flat_i.max() - flat_i.min()) * 0.2, 5.0)
            ax2.set_ylim(flat_i.min() - margin_i, flat_i.max() + margin_i)

    pdf_pages.savefig(fig1)
    pdf_pages.savefig(fig2)
    plt.close(fig1)
    plt.close(fig2)

    logger.info("-" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--data-dir", "-d",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing the CSV files to analyze.",
)
@click.option(
    "--pattern", "-p",
    default="*.csv",
    show_default=True,
    help="Glob pattern used to select CSV files inside --data-dir.",
)
@click.option(
    "--output-dir", "-o",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory to write output PDFs to. Defaults to --data-dir.",
)
@click.option(
    "--legend", "-l", "legends",
    multiple=True,
    help=(
        "Legend label(s), one per CSV file (in order). "
        "Repeat: -l 'Run 7' -l 'Run 8'. "
        "Defaults to the CSV filename stem."
    ),
)
@click.option(
    "--scale",
    type=click.Choice(["sec_scale", "min_scale", "hour_scale"]),
    default="sec_scale",
    help=(
        "Override the auto-detected time scale. "
        "By default the scale is chosen from the data span "
        "(< 5 min → sec, < 1 h → min, ≥ 1 h → hour)."
    ),
)
@click.option(
    "--overlay/--no-overlay",
    default=False,
    show_default=True,
    help="Also produce an overlay plot with all files on one canvas.",
)
@click.option(
    "--combined-pdf", "-c",
    default="power_supply_all.pdf",
    show_default=True,
    help="Filename for the combined multi-page PDF.",
)

@click.option("--verbose", "-v", is_flag=True, help="Verbose logging.")
def main(data_dir, output_dir, pattern, legends, scale, overlay,
         combined_pdf, verbose):
    """
    Plot power-supply voltage and current from every CSV in DIRECTORY.

    DIRECTORY can be a folder (all matching *.csv files are processed) or
    a single CSV file.

    \b
    Time axis is chosen automatically from the data span:
      span <  5 min   →  seconds
      span <  1 hour  →  minutes
      span >= 1 hour  →  hours
    Use --scale to override.

    \b
    Output per CSV:
      <stem>_voltage.pdf   individual voltage plot
      <stem>_current.pdf   individual current plot
    Plus a combined multi-page PDF collecting all plots.

    \b
    Examples:
      # Single file, custom legend
      python plot_power_supply.py data/run7.csv -l "Run 7: Φ=1.6×10¹⁰ p/cm²"

      # All CSVs in a folder, auto-legends, output to plots/
      python plot_power_supply.py data/ -o plots/

      # Override time scale, overlay comparison
      python plot_power_supply.py data/ --scale min_scale --overlay -l "Run 7" -l "Run 8"
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Collect CSV files
    data_dir = os.path.abspath(data_dir)
    if os.path.isfile(data_dir):
        csv_files = [data_dir]
        base_dir  = os.path.dirname(data_dir)
    else:
        csv_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        base_dir  = data_dir

    if not csv_files:
        raise click.ClickException(
            f"No files matching '{pattern}' found in: {data_dir}"
        )

    logger.info(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        logger.info(f"  {f}")

    # Output directory
    out_dir = os.path.abspath(output_dir) if output_dir else base_dir
    os.makedirs(out_dir, exist_ok=True)

    # Build legend list, pad with filenames when not provided
    legend_list = list(legends)
    for f in csv_files[len(legend_list):]:
        legend_list.append(os.path.splitext(os.path.basename(f))[0])

    if legends and len(legend_list) != len(csv_files):
        raise click.ClickException(
            f"Got {len(legends)} --legend value(s) but {len(csv_files)} CSV file(s). "
            "Provide one --legend per file, or none to use file names."
        )

    combined_path = os.path.join(out_dir, combined_pdf)
    with PdfPages(combined_path) as pdf_pages:
        if overlay:
            # All files on one shared canvas + individual files too
            plot_power_supply_parameters(
                csv_files=csv_files,
                legends=legend_list,
                pdf_pages=pdf_pages,
                scale_override=scale
            )
        else:
            # One canvas per file
            for data_file, lbl in zip(csv_files, legend_list):
                plot_power_supply_parameters(
                    csv_files=[data_file],
                    legends=[lbl],
                    pdf_pages=pdf_pages,
                    scale_override=scale
                )
    logger.info(f"Combined PDF → {combined_path}")
if __name__ == "__main__":
    main()
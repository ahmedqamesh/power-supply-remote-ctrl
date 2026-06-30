########################################################
"""
    Author: Ahmed Qamesh
    email: ahmed.qamesh@cern.ch
    Date: 29.08.2023 (refactored with click CLI)
"""
########################################################

import os
import glob
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
import click

from tests_lib.analysis_utils import AnalysisUtils
from tests_lib.logger_main import Logger
from tests_lib.plot_style import *  # noqa: F401,F403  (kept for project-wide plot styling)

os.environ["NUMEXPR_MAX_THREADS"] = "4"

log_format = '%(log_color)s[%(levelname)s]  - %(name)s -%(message)s'
log_call = Logger(log_format=log_format, name="Analysis", console_loglevel=logging.INFO, logger_file=False)
logger = log_call.setup_main_logger()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

TIME_LABELS = {
    "min_scale": "Time [Min]",
    "hour_scale": "Time [Hour]",
    "sec_scale": "Time [Sec]",
}

# ----------------------------------------------------------------------------
# Data analysis helpers
# ----------------------------------------------------------------------------
def calculate_efficiency_errors(Uout, Iout, Uin, Iin, eUout, eIout, eIin):
    """Compute efficiency, power, and their propagated errors."""
    efficiency = (Uout * Iout * 100) / (Uin * Iin)
    efficiency[efficiency.isnull()] = 0

    d_epsilon_d_Uout = (Iout * 100) / (Uin * Iin)
    d_epsilon_d_Iout = (Uout * 100) / (Uin * Iin)
    d_epsilon_d_Uin = (-Uout * Iout * 100) / (Uin ** 2 * Iin)
    d_epsilon_d_Iin = (-Uout * Iout * 100) / (Uin * Iin ** 2)

    delta_epsilon = np.sqrt(
        (d_epsilon_d_Uout * eUout) ** 2
        + (d_epsilon_d_Iout * eIout) ** 2
        + (d_epsilon_d_Uin * np.std(Uin)) ** 2
        + (d_epsilon_d_Iin * eIin) ** 2
    )

    power = Uout * Iout * 0.001
    delta_power = np.sqrt((Iout * eUout) ** 2 + (Uout * eIout) ** 2) * 0.001

    return efficiency, delta_epsilon, power, delta_power


def load_data(data_file):
    """Load a CSV file into a DataFrame, dropping a malformed trailing row if present."""
    data_frame = pd.read_csv(data_file, skiprows=[1])
    df_headers = data_frame.columns.tolist()
    data_frame = AnalysisUtils().check_last_row(data_frame=data_frame, column=df_headers[-1])
    return data_frame, df_headers


def should_use_raw_values(data_frame):
    """Decide whether to use raw samples instead of hourly/minute-binned averages.

    Falls back to raw values when there isn't enough of a time span to bin
    meaningfully (e.g. short runs spanning less than an hour).
    """
    if data_frame is None or data_frame.empty:
        return True

    timestamps = pd.to_datetime(data_frame['TimeStamp'], format='%Y-%m-%d_%H:%M:%S', errors='coerce').dropna()
    if timestamps.empty:
        return True

    span_seconds = (timestamps.max() - timestamps.min()).total_seconds()
    return span_seconds < 3600


def extract_data_info(data_file, min_scale=None):
    """Load a CSV and compute per-column averages (binned or raw)."""
    data_frame, df_headers = load_data(data_file)

    _, unique_days = AnalysisUtils().getDay(data_frame.TimeStamp)
    AnalysisUtils().getHours(TimeStamps=data_frame.TimeStamp, min_scale=min_scale, device="power_card")

    n_cols = len(df_headers)
    hourly_avg = [0] * n_cols
    hourly_std = [0] * n_cols
    new_headers = []

    use_raw_values = should_use_raw_values(data_frame)

    for pos, column in enumerate(df_headers):
        if 0 < pos < n_cols:
            if use_raw_values:
                values = pd.to_numeric(data_frame[column], errors='coerce').dropna().to_numpy()
                hourly_avg[pos - 1] = values
                hourly_std[pos - 1] = np.zeros_like(values, dtype=float)
            else:
                hourly_avg[pos - 1], hourly_std[pos - 1] = AnalysisUtils().get_hourly_average_value(
                    data_frame=data_frame,
                    column=column,
                    min_scale=min_scale,
                    unique_days=unique_days,
                )
            new_headers.append(column)

    return data_frame, new_headers, hourly_avg, hourly_std


def load_power_data(data_file, min_scale=None):
    """Extract elapsed time, supply voltage/current, and binned averages from a CSV."""
    data_frame, new_headers, hourly_avg, hourly_std = extract_data_info(data_file, min_scale=min_scale)

    f = 1000  # A -> mA
    elapsed = data_frame.iloc[:, 1].astype(float)

    _, unique_hours, unique_minutes = AnalysisUtils().getHours(
        TimeStamps=data_frame.TimeStamp, min_scale=min_scale, device="power_card"
    )

    if min_scale == "min_scale":
        period = np.arange(len(unique_minutes))
    else:
        period = np.arange(len(unique_hours))

    if len(data_frame) <= 3 or should_use_raw_values(data_frame):
        period = np.arange(len(data_frame))

    return elapsed, new_headers, hourly_avg, hourly_std, period


def _series(value, scale=1.0):
    """Coerce a hourly_avg entry (array or scalar list) into a flat float ndarray."""
    arr = np.asarray(value, dtype=float)
    return arr * scale


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_power_supply_parameters(csv_files, legends, min_scale, text_enable, pdf_pages, output_dir):
    """Plot supply voltage and current for each CSV file, overlaid on shared axes."""
    logger.info(f"Plotting test results for {len(csv_files)} file(s)")

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    time_label = TIME_LABELS.get(min_scale, "Time [Sec]")

    ax1.grid(True)
    ax1.set_ylabel("$V_{Supply}$ [V]")
    ax1.set_xlabel(time_label)
    ax1.autoscale(enable=True, axis='x', tight=None)
    if text_enable:
        ax1.set_title("Supply Voltage for the FPGA during Proton Irradiation")

    ax2.grid(True)
    ax2.set_ylabel("$I_{Supply}$ [mA]")
    ax2.set_xlabel(time_label)
    if text_enable:
        ax2.set_title("Supply Current for the FPGA during Proton Irradiation")

    last_legend_title = None
    all_voltages, all_currents = [], []

    for data_file, legend_title in zip(csv_files, legends):
        last_legend_title = legend_title
        logger.info(f"Processing: {data_file}")

        elapsed, headers, hourly_avg, hourly_std, period = load_power_data(data_file, min_scale=min_scale)

        voltage_series = _series(hourly_avg[1])
        current_series = _series(hourly_avg[5], scale=1000.0)
        x_values = np.arange(len(voltage_series)) if isinstance(hourly_avg[1], np.ndarray) else period

        if current_series.size > 0 and np.any(current_series != 0):
            delta_i = (np.max(current_series) - np.min(current_series)) / np.max(current_series) * 100
        else:
            delta_i = 0.0
        logger.info(f"Current Variation = {delta_i:.2f} %")

        ax1.plot(x_values, voltage_series, label=legend_title, marker='o')
        ax2.plot(x_values, current_series, label=legend_title, marker='o')
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        all_voltages.append(voltage_series)
        all_currents.append(current_series)

    if all_voltages:
        flat_v = np.concatenate(all_voltages)
        if flat_v.size > 0:
            ax1.set_ylim([np.min(flat_v) - 0.01, np.max(flat_v) + 0.01])
    if all_currents:
        flat_i = np.concatenate(all_currents)
        if flat_i.size > 0:
            ax2.set_ylim([np.min(flat_i) - 10, max(290, np.max(flat_i) + 10)])

    title_suffix = last_legend_title or ""

    ax1.legend(loc="upper left")
    ax1.set_title(f"Testing Results of the {title_suffix}")
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "power_supply_voltage.pdf"), bbox_inches='tight')
    pdf_pages.savefig(fig1)
    plt.close(fig1)

    ax2.legend(loc="upper left")
    ax2.set_title(f"Testing Results of the {title_suffix}")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "power_supply_current.pdf"), bbox_inches='tight')
    pdf_pages.savefig(fig2)
    plt.close(fig2)

    logger.info("-" * 100)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
@click.command()
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
    "--min-scale",
    type=click.Choice(["sec_scale", "min_scale", "hour_scale"]),
    default="sec_scale",
    show_default=True,
    help="Time binning scale used for averaging.",
)
@click.option(
    "--combined/--separate",
    default=False,
    show_default=True,
    help="Plot all CSVs overlaid on one pair of axes (combined) or one pair of axes per CSV (separate).",
)
@click.option(
    "--text-enable/--no-text-enable",
    default=False,
    show_default=True,
    help="Add titles to the generated plots.",
)
@click.option(
    "--legend", "-l", "legends",
    multiple=True,
    help="Custom legend label(s), one per file in the order files are processed. "
         "Defaults to the file name if not provided.",
)
def main(data_dir, pattern, output_dir, min_scale, combined, text_enable, legends):
    """Plot power-supply voltage/current curves for every CSV file in DATA_DIR."""
    output_dir = output_dir or data_dir
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not csv_files:
        raise click.ClickException(f"No files matching '{pattern}' found in {data_dir}")

    file_legends = list(legends) if legends else [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
    if len(file_legends) != len(csv_files):
        raise click.ClickException(
            f"Got {len(legends)} --legend value(s) but {len(csv_files)} CSV file(s); "
            "provide one legend per file, or none to use file names."
        )

    pdf_path = os.path.join(output_dir, "power_supply_parameters.pdf")
    pdf_pages = PdfPages(pdf_path)

    try:
        if combined:
            plot_power_supply_parameters(
                csv_files=csv_files,
                legends=file_legends,
                min_scale=min_scale,
                text_enable=text_enable,
                pdf_pages=pdf_pages,
                output_dir=output_dir,
            )
        else:
            for data_file, legend_title in zip(csv_files, file_legends):
                plot_power_supply_parameters(
                    csv_files=[data_file],
                    legends=[legend_title],
                    min_scale=min_scale,
                    text_enable=text_enable,
                    pdf_pages=pdf_pages,
                    output_dir=output_dir,
                )
    finally:
        pdf_pages.close()

    logger.info(f"Done. Combined PDF written to: {pdf_path}")


if __name__ == '__main__':
    main()
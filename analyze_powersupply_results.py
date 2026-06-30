########################################################
"""
    Author: Ahmed Qamesh (University of Wuppertal)
    email: ahmed.qamesh@cern.ch  
    Date: 29.08.2023
"""
########################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tests_lib.analysis_utils      import AnalysisUtils
from matplotlib.ticker import FormatStrFormatter
from tests_lib.logger_main   import Logger
from tests_lib.plotting   import Plotting
import logging
import pandas as pd
from scipy import interpolate

log_format = '%(log_color)s[%(levelname)s]  - %(name)s -%(message)s'
log_call = Logger(log_format=log_format, name="Analysis", console_loglevel=logging.INFO, logger_file=False)
logger = log_call.setup_main_logger()#

rootdir = os.path.dirname(os.path.abspath(__file__))
root_dir = rootdir + "/"
output_dir = "/data_files/"
import os
os.environ["NUMEXPR_MAX_THREADS"] = "4"  # Set the number of threads to 4
import numexpr as ne
from tests_lib.plot_style import *

text_enable = False

def calculate_efficiency_errors(Uout, Iout, Uin, Iin, eUout, eIout, eIin):
    # Calculate efficiency
    efficiency = (Uout * Iout * 100) / (Uin * Iin)
    efficiency[efficiency.isnull()] = 0  # Replace NaN values with 0
    # Calculate partial derivatives of efficiency with respect to each variable
    d_epsilon_d_Uout = (Iout * 100) / (Uin * Iin)
    d_epsilon_d_Iout = (Uout * 100) / (Uin * Iin)
    d_epsilon_d_Uin = (-Uout * Iout * 100) / (Uin ** 2 * Iin)
    d_epsilon_d_Iin = (-Uout * Iout * 100) / (Uin * Iin ** 2)
    
    # Calculate error on efficiency using error propagation
    delta_epsilon = np.sqrt(
        (d_epsilon_d_Uout * eUout)**2 +
        (d_epsilon_d_Iout * eIout)**2 +
        (d_epsilon_d_Uin * np.std(Uin))**2 +
        (d_epsilon_d_Iin * eIin)**2
    )
    
    # Calculate power
    power = Uout * Iout*0.001
    
    # Calculate error on power using error propagation
    delta_power = np.sqrt(
        (Iout * eUout)**2 +
        (Uout * eIout)**2
    )*0.001
    
    return efficiency, delta_epsilon, power, delta_power

def load_data(data_file = None,module_name = NotImplemented, file_name =None ):
    data_frame = pd.read_csv(data_file, skiprows=[1])
    logger.info(f"Analyzing {module_name} Data")    
    # logger.info the headers
    df_headers = data_frame.columns.tolist()
    data_frame = AnalysisUtils().check_last_row(data_frame = data_frame,column = df_headers[-1])
    return data_frame, df_headers

def should_use_raw_values(data_frame=None):
    if data_frame is None or data_frame.empty:
        return True

    timestamps = pd.to_datetime(data_frame['TimeStamp'], format='%Y-%m-%d_%H:%M:%S', errors='coerce')
    timestamps = timestamps.dropna()
    if timestamps.empty:
        return True

    span_seconds = (timestamps.max() - timestamps.min()).total_seconds()
    return span_seconds < 3600


def extract_data_info(data_file = None,module_name = None, file_name = None, min_scale = None):
    data_frame, df_headers = load_data(data_file = data_file, module_name = module_name, file_name =file_name)

    day, unique_days = AnalysisUtils().getDay(data_frame.TimeStamp)
    hours, unique_hours, unique_minutes = AnalysisUtils().getHours(TimeStamps=data_frame.TimeStamp, min_scale=min_scale, device="power_card")
    hourlyAverageValues = [0 for _ in range(len(df_headers))]
    hourlySTDValues = [0 for _ in range(len(df_headers))]
    new_headers = []

    use_raw_values = should_use_raw_values(data_frame)

    for pos, column in enumerate(df_headers):
        if pos > 0 and pos < len(df_headers):
            if use_raw_values:
                values = pd.to_numeric(data_frame[column], errors='coerce').dropna().to_numpy()
                hourlyAverageValues[pos-1] = values
                hourlySTDValues[pos-1] = np.zeros_like(values, dtype=float)
            else:
                hourlyAverageValues[pos-1], hourlySTDValues[pos-1] = AnalysisUtils().get_hourly_average_value(
                    data_frame=data_frame,
                    column=column,
                    min_scale=min_scale,
                    unique_days=unique_days,
                )

            new_headers = np.append(new_headers, column)

    return data_frame, new_headers, hourlyAverageValues, hourlySTDValues, day, unique_days

 
        
def extract_time_info(data_file = None,module_name = None, file_name = None, min_scale = None):
    data_frame, df_headers = load_data(data_file = data_file, module_name = module_name, file_name =file_name)
    #try: 
    day, unique_days = AnalysisUtils().getDay(data_frame.TimeStamp)
    #except: day = 0
    hours, unique_hours = AnalysisUtils().getHours(TimeStamps = data_frame.TimeStamp, min_scale =min_scale, device = "power_card")
    hourlyAverageValues = [0 for i in range(len(df_headers))]
    new_headers = []

    for pos,column in enumerate(df_headers):
        if pos > 0 and pos < len(df_headers):
            hourlyAverageValues[pos-1] = AnalysisUtils().getHourlyAverageValue(hours = hours,data = data_frame[column], min_scale = min_scale, unique_hours = unique_hours, unique_days = unique_days)   
            new_headers = np.append(new_headers, column)
    return data_frame, new_headers,hourlyAverageValues,day, unique_days
        
def load_power_data(data_file = None,module_name = None, file_name = None,min_scale = None):
    
    data_frame, new_headers,hourlyAverageValues, hourlySTDValues,day, unique_days = extract_data_info(data_file = data_file, module_name = module_name, file_name =file_name, min_scale = min_scale)

    f = 1000  # to convert to mA

    time = data_frame.iloc[:, 0].astype(str)
    elabsed = data_frame.iloc[:, 1].astype(float)

    Usin1 = data_frame.iloc[:, 2].astype(float)
    eUsin1 = data_frame.iloc[:, 3].astype(float)
    Iin1 = data_frame.iloc[:, 6].astype(float) * f
    eIin1 = data_frame.iloc[:, 7].astype(float) * f

    hours, unique_hours, unique_minutes = AnalysisUtils().getHours(TimeStamps=data_frame.TimeStamp, min_scale=min_scale, device="power_card")

    if min_scale == "min_scale":
        period = np.arange(len(unique_minutes))
    else:
        period = np.arange(len(unique_hours))

    if len(data_frame) <= 3 or should_use_raw_values(data_frame):
        period = np.arange(len(data_frame))

    return elabsed, Usin1, eUsin1, Iin1, eIin1, new_headers, hourlyAverageValues, hourlySTDValues, day, period


def plot_power_supply_parameters(modules_name=None,
                                 files_name=None, 
                                 text_enable =  None,
                                 legends =None,
                                 min_scale = None):
    logger.info(f"Plotting Test Results {modules_name}")
    fig1, ax1 = plt.subplots() 
    fig2, ax2 = plt.subplots() 

    if min_scale == "min_scale":
        time_label = "Time [Min]"
    elif min_scale == "hour_scale":
        time_label = "Time [Hour]"
    else:
        time_label = "Time [Sec]"
   
    ax1.grid(True)
    ax1.set_ylabel("$V_{Supply}$ [V]")#("Output Voltage $V_{out}$[V]")
    ax1.set_xlabel(time_label)#("Input Voltage $V_{in}$ [V]")
    ax1.autoscale(enable=True, axis='x', tight=None)
    if text_enable: ax1.set_title(f"Supply Voltage for the FPGA during Proton Irradiation")
    
    ax2.grid(True)
    ax2.set_ylabel("$I_{Supply}$ [mA]")#("Input Voltage $V_{in}$ [V]")
    ax2.set_xlabel(time_label)#(r'Current $I$ [mA]')
    if text_enable: ax2.set_title(f"Supply Current for the FPGA during Proton Irradiation")   

    for index, (file_name, module_name) in enumerate(zip(files_name, modules_name)):
        if legends: legend_title = legends[index]
        else: legend_title = module_name
        file_in_path = root_dir
        data_file = file_in_path + module_name + output_dir + file_name+".csv"
        # file_name_det = os.path.basename(data_file)
        # file_directory = os.path.dirname(data_file)
        if os.path.exists(data_file): 
            data_file = file_in_path + module_name + output_dir + file_name+".csv"
            file_out_path = root_dir
        else: 
            data_file = file_in_path + file_name+".csv"
            file_out_path = data_file[:-4]+".pdf"
           
        elabsed, Usin1, eUsin1, Iin1, eIin1, new_headers, hourlyAverageValues, hourlySTDValues, day, period = load_power_data(
            data_file=data_file,
            module_name=module_name,
            file_name=file_name,
            min_scale=min_scale,
        )

        if isinstance(hourlyAverageValues[1], np.ndarray):
            voltage_series = hourlyAverageValues[1]
            current_series = hourlyAverageValues[5] * 1000
            x_values = np.arange(len(voltage_series))
        else:
            voltage_series = np.asarray(hourlyAverageValues[1], dtype=float)
            current_series = np.asarray(hourlyAverageValues[5], dtype=float) * 1000
            x_values = period

        if np.size(current_series) > 0 and np.any(np.asarray(current_series) != 0):
            delta_I = (np.max(current_series) - np.min(current_series)) / np.max(current_series) * 100
        else:
            delta_I = 0.0

        logger.info(f"Output results in: {file_out_path}...")
        logger.report(f'Current Variation = {delta_I} %')

        ax1.plot(x_values, voltage_series, label=f"{legend_title}", marker='o')
        ax2.plot(x_values, current_series, label=f"{legend_title}", marker='o')
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    voltage_values = np.asarray(hourlyAverageValues[1], dtype=float)
    current_values = np.asarray(hourlyAverageValues[5], dtype=float) * 1000
    if voltage_values.size > 0:
        ax1.set_ylim([np.min(voltage_values) - 0.01, np.max(voltage_values) + 0.01])
    ax1.legend(loc="upper left")
    plt.tight_layout()
    fig1.savefig(file_out_path[:-4]+ "_voltage.pdf", bbox_inches='tight')   
    ax1.set_title(f"Testing Results of the {legend_title}")
    plt.tight_layout()
    PdfPages.savefig()  
    plt.close(fig1)
    
    ax2.set_ylim([min(hourlyAverageValues[5]*1000)-10, 290])    
    ax2.legend(loc="upper left") 
    plt.tight_layout()
    fig2.savefig(file_out_path[:-4]+ "_current.pdf", bbox_inches='tight')
    ax2.set_title(f"Testing Results of the {legend_title}")  
    plt.tight_layout()
    PdfPages.savefig()
    plt.close(fig2)
    print("--------------------------------------------------------------------------------------------------------")
# Main function
if __name__ == '__main__':
    # get program arguments
    PdfPages = PdfPages(root_dir+'power_supply_converters.pdf')
    modules_compare = ['output_dir/']
    files_compare =  ["output_dir/power_supply"]
    plot_power_supply_parameters(modules_name=modules_compare,
                                 files_name=files_compare,
                                 legends= ["Run 7: $\Phi = 1.6 x 10^{10}$ $Proton/cm^2$"],
                                  min_scale = "min_scale",
                                  text_enable = False)      
    

        
    PdfPages.close()
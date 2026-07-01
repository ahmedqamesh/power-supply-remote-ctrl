########################################################
"""
    Author: Ahmed Qamesh
    email: ahmed.qamesh@cern.ch  
    Date: 29.08.2023
"""
########################################################
import sys # For sys.argv and sys.exit
import numpy as np
import click
import yaml
import time
import os
import time
from datetime import datetime
import atexit
import tests_lib.power_supply_E36xxA_utils as E36xxA_lib
from tests_lib.analysis_utils      import AnalysisUtils
from tests_lib.logger_main   import Logger
import logging
log_format = '%(log_color)s[%(levelname)s]  - %(name)s -%(message)s'
log_call = Logger(log_format=log_format, name="Top", console_loglevel=logging.INFO, logger_file=False)
logger = log_call.setup_main_logger()#

rootdir = os.path.dirname(os.path.abspath(__file__))
timeout = 1
time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
output_dir = rootdir+"/output_dir/"+time_now

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def calculate_error(values = None, unit = "",output = None):
    mean_value = np.mean(values)
    std_dev = np.std(values)
    logger.report(f"Output {unit}[{output}]: {mean_value:.3f}+/-{std_dev:.3e}")
    return mean_value, std_dev

def exit_handler():
# This function will be called on script termination
    try:
        logger.warning("Closing the program.")
    except Exception:
        pass

def scan_supply_paramters(instrument = None,
                      outname = None,
                      output1=None,
                      output2=None,
                      voltage1 = None,
                      voltage2 = None,
                      num_samples = 3):
    logger.info(f"Scanning Supply Parameters...")
    file_headers =['TimeStamp','elabsed_time',"Usin1(V)","eUsin1(V)","Usin2(V)","eUsin2(V)","Isin1(A)","eIsin1(A)","Isin2(A)","eIsin2(A)"]
    ps_csv_writer, ps_csv_file = AnalysisUtils().build_data_base(fieldnames=file_headers, outputname = outname, directory=output_dir)        
    monitoringTime = time.time()
    i = 0
    # Register the termination signal handler
    atexit.register(exit_handler)
    try:
        while True:
            i = i+1
            if output1:
                voltage_mean_1, voltage_std_1,current_mean_1, current_std_1 = E36xxA_lib.set_nomianl_voltage(instrument = instrument, voltage= voltage1,  output=output1,num_samples = num_samples)
            else:
                voltage_mean_1, voltage_std_1,current_mean_1, current_std_1  = 0.0, 0.0, 0.0, 0.0
            if output2:
                voltage_mean_2, voltage_std_2,current_mean_2, current_std_2  = E36xxA_lib.set_nomianl_voltage(instrument = instrument,voltage= voltage2,output = output2,num_samples = num_samples) 
            else:
                voltage_mean_2, voltage_std_2,current_mean_2, current_std_2  = 0.0, 0.0, 0.0, 0.0 
            ts = time.time()
            file_time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            elapsedtime = ts - monitoringTime      
            ps_csv_writer.writerow((str(file_time_now),
                                    str(elapsedtime),
                                    str(voltage_mean_1),
                                    str(voltage_std_1),                                                    
                                    str(voltage_mean_2),
                                    str(voltage_std_2),   
                                    str(current_mean_1),
                                    str(current_std_1),                                                 
                                    str(current_mean_2),
                                    str(current_std_2)))       
            ps_csv_file.flush() # Flush the buffer to update the file
            time.sleep(timeout)
            print(f"--------------------------------------------------------------------")
    except (KeyboardInterrupt):
        #Handle Ctrl+C to gracefully exit the loop
        logger.warning("User interrupted")
        sys.exit(1)      
    finally:
        E36xxA_lib.set_nomianl_voltage(instrument = instrument, voltage= "0.0", output=output1,num_samples = num_samples)
        E36xxA_lib.set_nomianl_voltage(instrument = instrument,voltage= "0.0",output = output2,num_samples = num_samples) 
        ps_csv_writer.writerow((str(None),
                     str(None),
                     str(None),
                     str(None),
                     str(None),
                     str(None),
                     str(None),
                     str(None),
                     str(None), 
                     "End of Test"))  
    
        logger.info(f"Data are saved to {output_dir}/{outname}") 


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML configuration file"
)
@click.option(
    "--list",
    "list_devices",
    is_flag=True,
    default=False,
    help="List available power supply devices and exit"
)
@click.option(
    "--init",
    "init_device",
    is_flag=True,
    default=False,
    help="Initialize device only (no test execution)"
)
@click.option(
    "--set",
    "set_voltage",
    is_flag=True,
    default=False,
    help="Set device voltage (no test execution)"
)
@click.option(
    "--scan",
    "scan_parameters",
    is_flag=True,
    default=False,
    help="scan supply parameters (fully parameterized)"
)
@click.option(
    "--sample",
    "sample_parameters",
    is_flag=True,
    default=False,
    help="sample supply parameters (fully parameterized)"
)

def main(config, list_devices, init_device,set_voltage, scan_parameters, sample_parameters):
    cfg = load_config(config)

    # -------------------------
    # LIST MODE: list_available_devices
    # -------------------------
    if list_devices:
        E36xxA_lib.list_available_devices(
            msg=cfg["device"]["msg"],
            identification_to_check=cfg["device"]["identification_to_check"]
        )
        return
    # -------------------------
    # INIT MODE
    # -------------------------
    if init_device or set_voltage or scan_parameters or sample_parameters:

        instrument, rm, identification = E36xxA_lib.initialize_power_devices(
            check=False,
            resource=cfg["device"]["resource"]
        )

        out1, out2 = E36xxA_lib.get_device_outputs(
            identification=identification
        )
        logger.info(f"Device Initialized ")
        logger.info(f"-- Output mapping: {out1}, {out2}....")
        #E36xxA_lib.close_power_devices(instrument = instrument, rm = rm)    
        # -------------------------------------------------
        # 4. set_current_limit (multi-output)
        # -------------------------------------------------
        for _, cfg_out in cfg["outputs"].items():
            E36xxA_lib.set_current_limit(
                    instrument=instrument,
                    max_current=cfg_out["current_limit"],
                    output=cfg_out["name"]
                )
    
    if set_voltage:
        time.sleep(timeout)
        # -------------------------------------------------
        # 5. set_nomianl_voltage (voltage control)
        # -------------------------------------------------
        for _, cfg_out in cfg["outputs"].items():
            E36xxA_lib.set_nomianl_voltage(
                    instrument=instrument,
                    voltage=cfg_out["voltage"],
                    output=cfg_out["name"],
                    num_samples = cfg["test"]["num_samples"]
                )

    if scan_parameters:
        # -------------------------------------------------
        # 6. scan_supply_paramters (fully parameterized)
        # -------------------------------------------------
        output1 = cfg["outputs"]["out1"]["name"] if cfg["outputs"]["out1"]["active"] else None
        output2 = cfg["outputs"]["out2"]["name"] if cfg["outputs"]["out2"]["active"] else None
        scan_supply_paramters(
            instrument=instrument,
            outname=cfg["device"]["outname"],
            output1=output1,
            output2=output2,
            voltage1=cfg["outputs"]["out1"]["voltage"],
            voltage2=cfg["outputs"]["out2"]["voltage"],
            num_samples = cfg["test"]["num_samples"]
        )

    if sample_parameters:
        for _, cfg_out in cfg["outputs"].items():
            E36xxA_lib.sample_supply_paramters(instrument= instrument, num_samples = cfg["test"]["num_samples"]) 
if __name__ == "__main__":
    main()
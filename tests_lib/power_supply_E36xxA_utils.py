########################################################
"""
    Author: Ahmed Qamesh
    email: ahmed.qamesh@cern.ch  
    Date: 29.01.2022
"""
########################################################
#pip install pyvisa
#The script is dedicated for Agilent E364xA  https://www.tme.eu/Document/9f3996689f24777703b84ff64b156944/E3646-90001.pdf
import numpy as np
import time
import time
import pyvisa
import logging
from .logger_main   import Logger
log_format = '%(log_color)s[%(levelname)s]  - %(name)s -%(message)s'
log_call = Logger(log_format=log_format, name="E364xA Lib", console_loglevel=logging.INFO, logger_file=False)
logger = log_call.setup_main_logger()
visa_timeout = 1

def list_available_devices(msg = True, identification_to_check = "E3631A"):
        # Initialize the PyVISA library
        logger.report(f'Listing Available Devices...')
        rm = pyvisa.ResourceManager()
        # List available VISA resources
        resources = rm.list_resources()
        devices_dict = {}
        target_resource = None
        for resource in resources:
            #devices_return.append(resource)
            try:
                instrument = rm.open_resource(resource, timeout=10000,
                                              write_termination='\n',
                                              read_termination='\n')
                instrument.timeout = 10000  # Set a longer timeout (in milliseconds)
                idn_response = instrument.query('*IDN?')
                manufacturer = idn_response.split(',')[0]
                identification = idn_response.split(',')[1]
                devices_dict[identification] = resource
                if identification_to_check == "identification":
                    target_resource = resource
                else:
                    pass 
                if msg:  logger.info(f'Available Device    : {resource} - {manufacturer} - {identification}')
            except Exception as e:
                if msg:  logger.info(f'Available Resources : {resource}[E]')
                continue
        return devices_dict, target_resource
        
def initialize_power_devices(resource = None, check = None, identification_to_check = "E3631A"):
    if check:
        logger.notice(f'Checking Available Devices...')
        _, target_resource  = list_available_devices(msg=True,identification_to_check =  identification_to_check)
        resource_name = target_resource #devices_dict.get(identification)
    else: 
        resource_name = resource
    # Initialize the PyVISA library
    rm = pyvisa.ResourceManager()
    instrument = rm.open_resource(resource_name, timeout=5000,
                                  write_termination='\n',
                                  read_termination='\n')
    instrument.timeout = 5000
    idn_response = instrument.query('*IDN?')
    manufacturer = idn_response.split(',')[0]
    identification = idn_response.split(',')[1]
    logger.info(f'Available Device: {manufacturer} - {identification}')
    instrument.write('*RST') # Reset instrument
    instrument.write('SYSTem:REMote') # Reset instrument
    time.sleep(5)
    return instrument, rm ,identification

def close_power_devices(instrument = None, rm = None):
    # Close the resources
    instrument.close()
    rm.close()

def get_device_outputs(identification = None):
    if identification == "E3648A":
        out1,out2 = "OUTP1","OUTP2"
    elif identification == "E3631A":
        out1,out2 =  "P6V","P25V" 
    else: 
        out1,out2 =  "P6V","P25V" 
    return  out1,out2  

def set_current_limit(instrument = None, max_current = None, output=None):
    # Set the voltage and current for Output 1
    voltage_mean, voltage_std,current_mean, current_std = None, None,None, None
    logger.info(f"Apply current limit [{output}]: {max_current} A")
    idn_response = instrument.query('*IDN?')
    identification = idn_response.split(',')[1]
    
    if identification  == "E3631A":
        pass
    
    elif identification == "E3648A":
        instrument.write(f'INST:SEL {output}')
        time.sleep(0.1)
        instrument.write(f'CURRent {max_current}')
        time.sleep(0.1)
    
    else: 
        pass
    instrument.write(f'OUTP ON')    
    time.sleep(0.2)
    return None     

def set_nomianl_voltage(instrument = None,
                        voltage= None,
                        output=None,
                        num_samples=None):
    
    voltage_mean, voltage_std,current_mean, current_std = None, None,None, None
    logger.info(f"Apply Output Voltage[{output}]: {voltage} V")
    idn_response = instrument.query('*IDN?')
    identification = idn_response.split(',')[1]

    if identification  == "E3631A":
        instrument.write(f'APPLy {output} ,{voltage}')
    
    elif identification == "E3648A":
        instrument.write(f'INST:SEL {output}')
        instrument.write(f'VOLT {voltage}')
        
    else: 
        instrument.write(f'APPLy {output} ,{voltage}')
    instrument.write(f'OUTP ON')    
    voltage_mean, voltage_std,current_mean, current_std  = sample_supply_paramters(instrument= instrument,num_samples = num_samples) 
    return voltage_mean, voltage_std,current_mean, current_std      

def sample_supply_paramters(instrument = None,num_samples = None):    
    voltage_values = []
    current_values = []
    time.sleep(visa_timeout)  
    logger.info(f"Sample Supply Parameters...")
    #error_message = instrument.query('SYST:ERR?')
    _output = instrument.query("INST:SEL?").split() #check which output
    instrument.write(f'OUTP ON') 
    time.sleep(visa_timeout)  
    for i in range(num_samples): 
        voltage_output= float(instrument.query("MEAS:VOLT?"))
        current_output = float(instrument.query("MEAS:CURR?"))
        #APPLy_output = instrument.query("APPLy?")
        voltage_values.append(float(voltage_output))
        current_values.append(float(current_output))
        
    voltage_mean, voltage_std = calculate_error(values = voltage_values,unit = "V",output = _output)
    current_mean, current_std = calculate_error(values = current_values,unit =  "A", output = _output)
    return voltage_mean, voltage_std,current_mean, current_std

         
def get_power_source_parameters(instrument = None, output=None): 
    current_output_mode = instrument.query('INST?').strip()
    #instrument.write(f'OUT{output}')  # Select Output 
    voltage_output = float(instrument.query('VOLT?'))
    logger.info(f"Voltage Output {output}[{current_output_mode}]: {voltage_output} V")
    # Read back the current for Output 
    #instrument.write(f'OUTP ON')          #enable the output 
    current_output = float(instrument.query('CURR?'))
    logger.info(f"Current Output {output}[{current_output_mode}]: {current_output:.3f}A")
    # current_max = float(instrument.query('CURR? MAX'))
    # current_min = float(instrument.query('CURR? MIN'))
    time.sleep(visa_timeout)
    return voltage_output,current_output


def calculate_error(values = None, unit = "",output = None):
    mean_value = np.mean(values)
    std_dev = np.std(values)
    if unit == "A": parameter = "Current"
    else: parameter = "Voltage"
    logger.report(f"Output {parameter}{output}: {mean_value:.3f}+/-{std_dev:.3e} {unit}")
    return mean_value, std_dev

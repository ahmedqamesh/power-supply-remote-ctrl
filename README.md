# Power Supply Remote Control (PyVISA + Click + YAML)

A Python-based CLI tool for controlling and testing programmable power supplies (e.g., Keysight/Agilent E36xxA series) via PyVISA.  
The system is fully configurable using YAML and supports device discovery, initialization, voltage/current configuration, and automated test execution.

---
## Features

- 🔌 VISA-based instrument communication (PyVISA)
- ⚙️ YAML-driven configuration (no hardcoded parameters)
- 🧪 Modular execution modes (list, init, configure, scan, sample)
- ⚡ Multi-output support (e.g., out1 / out2)
- 📊 Automated parameter scanning and sampling
- 🧰 Click-based CLI interface

---

## Project Structure
```bash
.
├── configs/ # YAML configuration files
├── tests_lib/ # Instrument control library (SCPI wrappers)
├── test_power_supply.py # Main CLI entry point
├── analyze_powersupply_results.py
├── output_dir/ # Generated logs/results
├── requirements.txt
└── README.md
```
---

## Installation

1. Create environment
```bash
python -m venv env
source env/bin/activate
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

###Configuration (YAML)


Example configs/ps_E3631A.yaml:
```yaml
device:
  msg: true
  identification_to_check: "E3631A"
  resource: "ASRL/dev/ttyUSB1::INSTR"

outputs:
  out1:
    name: "OUT1"
    voltage: "5.0"
    current_limit: "1.0"
    active: true

  out2:
    name: "OUT2"
    voltage: "0.0"
    current_limit: "0.5"
    active: true

test:
  num_samples: 5
  power_name: "power_supply"
```

# CLI Usage
```bash
# Step 1: detect instruments
python test_power_supply.py --config configs/ps.yaml --list

# Step 2: initialize device
python test_power_supply.py --config configs/ps.yaml --init

# Step 3: configure voltage
python test_power_supply.py --config configs/ps.yaml --set

# Step 4: run scan
python test_power_supply.py --config configs/ps.yaml --scan

```
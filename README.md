# Supplementary Materials for:
## Computational Configuration of a Multi-Channel Confined-Space Gas Detector

### Authors
shaole Yu;wenying Zhang

### Overview
This repository contains the code and supplementary data for the paper published in IEEE Sensors Journal.

### Contents
| File | Description |
|------|-------------|
| `run_ieee_experiments.py` | Main experiment runner for 20-seed comparisons |
| `gas_optimization_experiment.py` | SC-MONRBO optimizer implementation |
| `gas_sensitivity.py` | Sensitivity analysis and ablation scripts |
| `code_guided_scan_analysis.py` | Asynchronous ZOH stress test analysis |
| `physical_validation_data_template.csv` | Template for Stage III physical validation |
| `power_measurement_data_template.csv` | Template for power measurement data |
| `embedded_resource_data_template.csv` | Template for embedded resource data |
| `source_manifest.csv` | File manifest with checksums |

### Requirements
- Python 3.8+
- NumPy, SciPy, Matplotlib, Pandas

### License
MIT License

### Citation
If you use this code, please cite the accompanying paper:
Safety-Constrained Multi-Objective Configuration of Gas Detector Parameters: A Reproducible Computational Screening Framework

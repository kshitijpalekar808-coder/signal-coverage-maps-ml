# Sample Drive-Test Datasets
**MathWorks Excellence in Innovation — Project #151**

This directory contains lightweight sample datasets designed for immediate testing without downloading or processing large datasets, fulfilling MathWorks Project Repository Guideline 3 (*"Include a small sample dataset in data/sample/ sufficient for a basic test without extra downloads"*).

---

### Files Included

1. `sample_urban_grid.csv` (100 rows, 14 KB):
   - Extracted from the synthetic Manhattan-grid urban microcell simulation.
   - Includes GPS coordinates (`Latitude`, `Longitude`), route labels (`RouteID`), base station position (`Serving_BS_Lat`, `Serving_BS_Lon`), frequency (2100 MHz), and signal quality indicators (`RSRP_dBm`, `RSRQ_dB`, `SINR_dB`).

2. `sample_real_mysignals.csv` (100 rows, 16 KB):
   - Extracted from the authentic field telemetry dataset collected by the Wireless Communications Laboratory (WCL) at Technical University of Crete (mySignals GSM-1800).
   - Includes real cellular measurements (`Timestamp`, `Latitude`, `Longitude`, `X`, `Y`, `Distance`, `LogDist`, `Azimuth`, `RouteID`, `CellID`, `Frequency_MHz`, `RSRP_dBm`).

---

### Usage in MATLAB
```matlab
% Read sample dataset
sampleData = readtable(fullfile('data', 'sample', 'sample_urban_grid.csv'));
disp(head(sampleData, 5));
```

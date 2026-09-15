# Dataset Provenance & Schema Documentation

This repository employs a **dual-dataset evaluation framework** designed to balance real-world empirical validation with controlled theoretical rigor:

1. **`real_mysignals_dataset.csv`**: Authentic real-world GSM-1800 field drive-test telemetry from Chania, Greece. Used for the primary real-world model benchmark.
2. **`synthetic_urban_grid_dataset.csv`**: Controlled physics-informed synthetic Manhattan-grid drive-test dataset. Used for experiments where continuous 2D ground truth across every square meter is strictly required (ground-truth coverage error mapping, 2%–40% sparsity sweep, and 20-trial active learning trajectory comparison).

---

## 1. Genuine Real-World Field Dataset: `real_mysignals_dataset.csv`

* **Source:** **mySignals Project** (Alimpertis, Fasarakis-Hilliard, & Bletsas, *"Community RF Sensing for Source Localization"*, IEEE Wireless Communications Letters, 2014).
* **Official Challenge Reference:** Direct recommendation from the official [MathWorks Challenge #151 brief](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/tree/main/projects/Signal%20Coverage%20Maps%20Using%20Measurements%20and%20Machine%20Learning).
* **Environment:** Urban Chania, Crete, Greece.
* **Carrier Frequency:** $1860.2\text{ MHz}$ (GSM-1800 Downlink, ARFCN 787).
* **Base Transceiver Station (BTS):** Macrocell sector `6056x` located at $(35.508354^\circ\text{N}, 24.024523^\circ\text{E})$.
* **Telemetry Samples:** 898 spatial grid-thinned road measurement points collected by moving mobile sensors with high-accuracy GPS ($\le 65\text{ m}$).
* **Spatial Holdout Strategy (Zero Data Leakage):**
  - Divided into 4 balanced geographic sectors across the city based on coordinate medians:
    - **Zone 1 (Southwest Corridor):** 236 samples ($26.3\%$)
    - **Zone 2 (Northwest Corridor):** 213 samples ($23.7\%$)
    - **Zone 3 (Southeast Corridor):** 213 samples ($23.7\%$)
    - **Zone 4 (Northeast Held-Out Corridor):** 236 samples ($26.3\%$)
  - **Training Set:** Zones 1, 2, 3 ($N = 662$, $73.7\%$)
  - **Testing Set:** Zone 4 ($N = 236$, $26.3\%$)
  - **Leakage Policy:** $\text{Zone}_{\text{train}} \cap \text{Zone}_{\text{test}} = \emptyset$.

### Column Schema (`real_mysignals_dataset.csv`):
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Timestamp` | String | Timestamp of mobile measurement |
| `Latitude` | Float64 | Geodetic WGS-84 Latitude (degrees) |
| `Longitude` | Float64 | Geodetic WGS-84 Longitude (degrees) |
| `X` | Float64 | Local Cartesian metric Easting relative to BTS (m) |
| `Y` | Float64 | Local Cartesian metric Northing relative to BTS (m) |
| `Distance` | Float64 | 2D Euclidean distance to BTS: $\sqrt{X^2 + Y^2}$ (m) |
| `LogDist` | Float64 | $\log_{10}(\text{Distance})$ |
| `Azimuth` | Float64 | Bearing angle relative to BTS: $\text{atan2d}(Y, X)$ (degrees) |
| `RouteID` | Integer | Spatial Zone Identifier (1 to 4) |
| `CellID` | Integer | Serving Cell Identifier (e.g. 60562, 60564, 60565) |
| `Frequency_MHz` | Float64 | Downlink Carrier Frequency (1860.2 MHz) |
| `RSRP_dBm` | Float64 | Received Signal Strength Indicator (dBm, Target variable) |

---

## 2. Controlled Synthetic Urban Grid Dataset: `synthetic_urban_grid_dataset.csv`

* **Nature:** **Physics-Informed Synthetic Urban Drive-Test Dataset**
* **Environment:** Urban Manhattan-grid cellular propagation environment ($1000\text{ m} \times 1000\text{ m}$) operating at $2.1\text{ GHz}$.
* **Propagation Physics:** Generated using a simplified single-slope UMi path-loss formula ($PL(d) = 32.4 + 20\log_{10}(f) + 30\log_{10}(d)$) with multi-scale correlated building shadow fading and Gaussian multipath fast fading.
* **Serving Base Station:** Centrally positioned at $(500\text{ m}, 500\text{ m})$ local Cartesian space ($42.3601^\circ\text{N}, -71.0942^\circ\text{W}$) with $43\text{ dBm}$ (20 W) transmit power and $15\text{ dBi}$ sector antenna gain.
* **Trajectory Geometry (12 Street Routes):**
  - **Routes 1–6 (Horizontal Avenues):** Driven along $y \in \{-400, -250, -100, 50, 200, 350\}\text{ m}$ relative to BS.
  - **Routes 7–12 (Vertical Cross-Streets):** Driven along $x \in \{-400, -250, -100, 50, 200, 350\}\text{ m}$ relative to BS.
* **Spatial Holdout:** Routes 1–9 train ($N = 1017$, $75.0\%$), Routes 10–12 test ($N = 339$, $25.0\%$).
* **Role in Research:** Reserved strictly for controlled experiments where continuous 2D ground truth ($N = 10,201$ grid nodes) is required to evaluate true interpolation errors across unmeasured terrain and simulate adaptive drive-test routing.

### Column Schema (`synthetic_urban_grid_dataset.csv`):
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Timestamp` | DateTime | Measurement timestamp (2-second interval) |
| `Latitude` | Float64 | Geodetic Latitude (degrees) |
| `Longitude` | Float64 | Geodetic Longitude (degrees) |
| `RouteID` | Integer | Street Route Identifier (1 to 12) |
| `Serving_BS_Lat`| Float64 | Serving Base Station Latitude |
| `Serving_BS_Lon`| Float64 | Serving Base Station Longitude |
| `PCI` | Integer | Physical Cell Identity (101) |
| `Frequency_MHz` | Integer | Carrier Frequency (2100 MHz) |
| `RSRP_dBm` | Float64 | Received Signal Power (dBm, Target variable) |
| `RSRQ_dB` | Float64 | Reference Signal Received Quality (dB) |
| `SINR_dB` | Float64 | Signal-to-Interference-plus-Noise Ratio (dB) |

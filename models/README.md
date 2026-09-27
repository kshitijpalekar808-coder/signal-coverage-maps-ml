# Pre-Trained Machine Learning Models
**MathWorks Excellence in Innovation — Project #151**

This directory contains pre-trained machine learning model artifacts generated from drive-test measurements, fulfilling MathWorks Project Repository Guideline 4 (*"Include trained model files in models/. Ensure the main script loads and runs the pre-trained model directly"*).

---

### Artifact Inventory & Native Architecture

All model artifacts are stored as native MATLAB structures trained using high-performance, vectorized core MATLAB routines. They require **zero proprietary or paid toolboxes** (no Statistics and Machine Learning Toolbox required for either loading or inference).

| Model Artifact | Algorithm & Implementation | Input Features | Output Target | Internal Fields & Native Architecture |
|---|---|---|---|---|
| `gpr_spatial_model.mat` | Spatial ARD Gaussian Process Regression (Native Matérn 5/2 Covariance Engine) | $[X, Y]$ normalized | $\text{RSRP}$ (dBm) | Struct `gprModel` containing learned length-scale vector `ell_vec`, signal standard deviation `sigma_f`, noise standard deviation `sigma_n`, Cholesky factor `L`, dual weight vector `alpha`, regularization `jitter`, training normalization parameters (`mu_X`, `std_X`, `mu_y`, `std_y`), and inference closure function (`predict`). Hyperparameters optimized via Nelder-Mead simplex MLL maximization (`fminsearch`) on training corridors 1–9. |
| `random_forest_model.mat` | Bagged Ensemble of Regression Trees (Native Pure-MATLAB Ensemble) | $[X, Y, d, \log_{10}(d), \phi]$ | $\text{RSRP}$ (dBm) | Struct `rfModel` containing native ensemble representation (`.native`) with individual binary decision tree structures (split features, cut points, leaf outputs), out-of-bag importance scores (`OOBPermutedPredictorImportance`), feature names, and inference closure function (`predict`). |
| `log_distance_model.mat` | Empirical Log-Distance Path Loss (Native OLS Matrix Inversion) | $\log_{10}(d)$ | $\text{RSRP}$ (dBm) | Struct `logModel` containing regression coefficient vector `beta` solved via core MATLAB backslash operator (`X \ y`), reference intercept `intercept`, slope `slope`, estimated path loss exponent `pathLossExponent` ($n = -\text{slope}/10$), and inference closure function (`predict`). |

---

### Loading and Inference Usage (MATLAB)

All pre-trained artifacts run directly in base MATLAB (R2021a or newer) with zero external toolbox dependencies.

#### 1. Spatial ARD Gaussian Process Regression
```matlab
% Load pre-trained spatial GPR model
data = load(fullfile('models', 'gpr_spatial_model.mat'));
gprModel = data.gprModel;

% Predict RSRP, latent spatial uncertainty, and observation uncertainty
% queryTable must contain columns 'X' and 'Y' (or pass Nx2 coordinate matrix)
[predRSRP, sigmaLatent, sigmaObs] = gprModel.predict(queryTable);

fprintf('Mean predicted RSRP: %.2f dBm (Avg Latent Uncertainty: %.2f dB)\n', ...
    mean(predRSRP), mean(sigmaLatent));
```

#### 2. Random Forest Regression Ensemble
```matlab
% Load pre-trained Random Forest model
data = load(fullfile('models', 'random_forest_model.mat'));
rfModel = data.rfModel;

% Predict RSRP on test features [X, Y, Distance, LogDist, Azimuth]
predRSRP_rf = rfModel.predict(testTable);
```

#### 3. Empirical Log-Distance Model
```matlab
% Load pre-trained Log-Distance path loss model
data = load(fullfile('models', 'log_distance_model.mat'));
logModel = data.logModel;

% Predict RSRP from distance features or coordinate table
predRSRP_log = logModel.predict(testTable);
fprintf('Path Loss Exponent n: %.2f, Reference Intercept: %.2f dBm\n', ...
    logModel.pathLossExponent, logModel.intercept);
```

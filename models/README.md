# Pre-Trained Machine Learning Models
**MathWorks Excellence in Innovation — Project #151**

This directory contains pre-trained machine learning model artifacts generated from drive-test measurements, fulfilling MathWorks Project Repository Guideline 4 (*"Include trained model files in models/. Ensure the main script loads and runs the pre-trained model directly"*).

---

### Artifact Inventory

| Model Artifact | Algorithm | Input Features | Target | Description |
|---|---|---|---|---|
| `gpr_spatial_model.mat` | Spatial ARD Gaussian Process Regression (`fitrgp`) | $[X, Y]$ normalized | $\text{RSRP}$ (dBm) | Proposed anisotropic Matérn 5/2 GPR kernel with closed-form uncertainty quantification. Trained on corridors 1–9. |
| `random_forest_model.mat` | Bagged Ensemble of 100 Regression Trees (`TreeBagger`) | $[X, Y, d, \log_{10}(d), \phi]$ | $\text{RSRP}$ (dBm) | High-capacity ensemble baseline capturing non-linear interactions across spatial coordinates and physics features. |
| `log_distance_model.mat` | Empirical Log-Distance Path Loss (`fitlm`) | $\log_{10}(d)$ | $\text{RSRP}$ (dBm) | Benchmark baseline estimating path-loss exponent $n$ via ordinary least squares. |

---

### Loading and Inference Usage (MATLAB)

```matlab
% Load pre-trained spatial GPR model
data = load(fullfile('models', 'gpr_spatial_model.mat'));
gprModel = data.gprModel;

% Predict RSRP and analytical standard deviation on new query points
% queryTable must contain columns 'X' and 'Y'
[predRSRP, sigmaLatent, yCI] = gprModel.predict(queryTable);

fprintf('Mean predicted RSRP: %.2f dBm (Avg Std: %.2f dB)\n', mean(predRSRP), mean(sigmaLatent));
```

```matlab
% Load pre-trained Random Forest model
data = load(fullfile('models', 'random_forest_model.mat'));
rfModel = data.rfModel;

% Predict RSRP on test features [X, Y, Distance, LogDist, Azimuth]
predRSRP_rf = rfModel.predict(testTable);
```

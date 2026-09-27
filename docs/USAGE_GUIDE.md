# Evaluator and User Guide
## Signal Coverage Maps Using Measurements and Machine Learning
**MathWorks Excellence in Innovation — Project #151**

This guide provides step-by-step instructions for running, testing, and evaluating the repository with minimal manual setup.

---

### 1. Prerequisites and System Requirements
- **MATLAB Release:** R2022b or later (recommended R2023a / R2024a / R2024b)
- **Operating System:** Windows, macOS, or Linux
- **Required MATLAB Toolboxes:**
  - Statistics and Machine Learning Toolbox (for GPR, TreeBagger Random Forest, evaluation metrics)
  - Mapping Toolbox *(Optional / Fallback provided)*: Automated Haversine / UTM projections fallback gracefully if Mapping Toolbox is absent.
  - Parallel Computing Toolbox *(Optional)*: Accelerates cross-validation and active learning trials.

---

### 2. Quick Verification (5 Seconds)
To verify that the environment, models, and data pipeline function properly without running the full benchmark:

```matlab
% In MATLAB Command Window:
quick_demo
```
This loads pre-trained models from `models/` and sample measurements from `data/sample/`, runs inference, outputs RMSE/MAE metrics, and produces a rapid verification figure.

---

### 3. Running Automated Integrity Tests (Reproducibility Audit)
To execute the automated 12-test scientific verification suite:

```matlab
validate_project
```
This verifies:
1. Real mySignals dataset schema and data types
2. 4-Zone spatial partitioning
3. Strict zero-leakage corridor holdouts
4. Preprocessing and physics-based feature extraction
5. Theoretical 3GPP path-loss bounds
6. Log-distance OLS monotonicity
7. Spatial ARD GPR hyperparameter optimization
8. Uncertainty quantification bounds
9. Random Forest non-negativity
10. Active learning variance reduction
11. Synthetic grid consistency
12. End-to-end benchmark reproducibility

---

### 4. Running the Complete System End-to-End
To run the complete benchmark suite, ablation study, active learning experiments, and map generation:

```matlab
main
```
Or directly:
```matlab
run_all_experiments
```

**Outputs Produced:**
- Real-world mySignals GSM benchmark table (`results/real_field_benchmark_results.csv`)
- Synthetic grid benchmark table (`results/benchmark_results.csv`)
- High-resolution 300 DPI publication figures in `results/figures/`:
  - `real_field_coverage_reconstruction.png`
  - `spatial_coverage_reconstruction.png`
  - `active_learning_curve.png`
  - `feature_ablation_comparison.png`
  - `sparsity_sweep_analysis.png`

---

### 5. Running Standalone Modules
- `run_real_field_benchmark`: Runs the authentic field telemetry evaluation on mySignals GSM-1800 data.
- `run_ablation_study`: Executes the feature ablation study (Spatial vs. Physics vs. Hybrid).
- `run_sparsity_test`: Runs sparsity sweeps from 2% to 40% sampling density.
- `run_active_learning`: Evaluates route-based active learning drive-test optimization across 20 trials.

---

### 6. Repository Structure
```
├── LICENSE                       <- Open-source BSD 2-Clause License
├── README.md                     <- Primary project documentation
├── TOOLBOXES.md                  <- Toolbox dependency declarations
├── main.m                        <- Master entry point (One-Click Run)
├── quick_demo.m                  <- 5-second fast verification script
├── validate_project.m            <- Automated 12-test reproducibility suite
├── run_all_experiments.m         <- Full experimental runner
├── data/
│   ├── sample/                   <- Small standalone sample datasets for basic testing
│   ├── synthetic_urban_grid_dataset.csv
│   └── real_mysignals_dataset.csv
├── models/                       <- Pre-trained machine learning model artifacts (.mat)
├── docs/                         <- Project report, architecture, and guides
├── results/                      <- Benchmark CSV metrics and exported figures
├── src/                          <- Modular pipeline source code
│   ├── active_learning/          <- Variance-guided route selection
│   ├── data/                     <- Data ingestion and spatial partitioning
│   ├── evaluation/               <- Error metrics and ablation analysis
│   ├── models/                   <- GPR, Random Forest, IDW, Log-Distance, 3GPP
│   ├── preprocessing/            <- Coordinate transforms and physics features
│   └── visualization/            <- Map rendering, uncertainty plots, and heatmaps
└── tests/                        <- MATLAB unit test suite
```

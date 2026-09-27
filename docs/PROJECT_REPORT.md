# Engineering and Scientific Report
## Signal Coverage Maps Using Measurements and Machine Learning
**MathWorks Excellence in Innovation — Project #151**  
**Author:** Kshitij Palekar  
**Institution:** Vidyalankar Institute of Technology  
**Contact:** kshitij.palekar@vit.edu.in  
**Repository:** [github.com/kshitijpalekar808-coder/signal-coverage-maps-ml](https://github.com/kshitijpalekar808-coder/signal-coverage-maps-ml)  
**License:** BSD 2-Clause License  

---

### Executive Summary
Radio frequency (RF) signal coverage mapping is fundamental for cellular planning, base station siting, interference mitigation, and autonomous vehicle connectivity. Traditional ray-tracing simulations require prohibitive compute and high-fidelity 3D building databases, while empirical path-loss formulas (e.g., 3GPP TR 38.901 / Okumura-Hata) fail to resolve street canyon shadowing and diffraction effects.

This project delivers a **reproducible, leak-free machine learning pipeline** implemented entirely in MATLAB that reconstructs continuous, high-resolution cellular signal coverage maps (RSRP) from sparse drive-test measurements. We benchmark spatial interpolation methods, tree ensembles, and Gaussian Process Regression (GPR) across both synthetic urban Manhattan-grid topologies and authentic real-world drive-test telemetry (mySignals GSM-1800). Furthermore, we demonstrate a **route-aware active learning** optimization that reduces required drive-test mileage by 50% while accelerating map convergence.

---

### Key Technical Contributions
1. **Zero Spatial-Leakage Validation Protocol**:
   - Random $k$-fold cross-validation inflates RSRP predictive performance due to strong spatial autocorrelation. We enforce strict **spatial corridor holdouts** (synthetic routes 10–12 and real field Zone 4 holdouts), guaranteeing that test measurements share zero street segments with training observations.
2. **Feature Collinearity & Ablation Diagnosis**:
   - Rigorous feature ablation reveals that combining raw Cartesian coordinates $[X, Y]$ with explicit radio physics features $[\text{Distance}, \log_{10}(\text{Distance}), \text{Azimuth}]$ causes numerical ill-conditioning in anisotropic GPR covariance kernels ($\sigma_f^2 \exp(-\frac{1}{2} (x-x')^T \Lambda (x-x'))$).
   - A pure spatial ARD Matérn 5/2 GPR achieves superior generalization (synthetic RMSE: $2.49\text{ dB}$, real field RMSE: $4.49\text{ dB}$) without redundant distance priors.
3. **Rigorous Physical Baseline Grounding**:
   - Benchmarked against:
     - 3GPP TR 38.901 Urban Micro (UMi) Line-of-Sight/Non-Line-of-Sight propagation.
     - Empirical Log-Distance Path Loss (Ordinary Least Squares).
     - Inverse Distance Weighting ($k\text{NN-IDW}, p=2, k=30$).
     - Natural Neighbor Voronoi-based Spatial Interpolation.
     - Tree-Bagger Random Forest Regressor (150 trees).
4. **Bayesian Spatial Uncertainty Mapping**:
   - GPR yields closed-form analytical posterior variance $\sigma^2(x_*)$, producing pixel-wise 95% confidence intervals and uncertainty topographies across unmeasured city blocks.
5. **Route-Aware Active Learning**:
   - Iteratively selects next drive-test routes that maximize integrated predictive variance (entropy sampling), demonstrating up to $50\%$ reduction in drive-test surveying overhead.

---

### Comparative Benchmark Results

#### 1. Real-World Field Telemetry (mySignals GSM-1800, Held-Out Spatial Zone 4)
| Model | RMSE (dB) | MAE (dB) | $R^2$ Score | Max AE (dB) | P95 AE (dB) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Spatial ARD GPR (Proposed)** | **4.49** | **3.67** | **0.865** | **11.20** | **7.81** |
| Random Forest (150 Trees) | 5.21 | 4.31 | 0.818 | 13.45 | 9.40 |
| Natural Neighbor Interpolation | 6.84 | 5.62 | 0.686 | 17.80 | 12.50 |
| kNN-IDW ($p=2, k=30$) | 7.92 | 6.45 | 0.580 | 20.15 | 14.30 |
| Empirical Log-Distance (OLS) | 9.15 | 7.40 | 0.440 | 23.50 | 16.80 |

#### 2. Synthetic Controlled Grid Benchmark (Held-Out Corridors 10–12)
| Model | RMSE (dB) | MAE (dB) | $R^2$ Score | Max AE (dB) | P95 AE (dB) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Spatial ARD GPR (Proposed)** | **2.49** | **1.98** | **0.958** | **6.82** | **4.71** |
| Random Forest (150 Trees) | 3.12 | 2.45 | 0.934 | 8.90 | 6.15 |
| Natural Neighbor Interpolation | 4.78 | 3.82 | 0.845 | 13.20 | 9.40 |
| kNN-IDW ($p=2, k=30$) | 5.85 | 4.67 | 0.768 | 15.60 | 11.25 |
| Empirical Log-Distance (OLS) | 8.42 | 6.78 | 0.521 | 21.40 | 15.60 |
| 3GPP-Inspired UMi (No Training)| 11.85 | 9.60 | 0.051 | 28.70 | 22.10 |

---

### System Architecture and Reproducibility
- **Single Master Entry Point:** `main.m` executes the complete scientific pipeline without manual configuration.
- **Verification Suite:** `validate_project.m` executes an automated 12-stage validation suite checking schema integrity, zero spatial leakage, hyperparameter bounds, and reproducibility.
- **Publication Graphics:** Automatic generation of 300 DPI figures illustrating predicted coverage heatmaps, drive-test trajectories, variance surfaces, and error residuals.

---

### Conclusion
By enforcing rigorous spatial corridor partitioning, diagnosing kernel ill-conditioning, and leveraging Gaussian Process Regression, the proposed pipeline achieves state-of-the-art radio coverage mapping with bounded uncertainty and 50% drive-test route reduction via active learning.

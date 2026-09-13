# MATLAB Toolbox Dependencies & Requirements

**MathWorks Excellence in Innovation — Project #151**

## 🧩 Toolbox Compatibility Matrix

| Component | Required Toolboxes | Native Fallback Engine Provided? | Compatibility |
| :--- | :--- | :---: | :---: |
| **Core Pipeline Execution** | Base MATLAB (R2021a or newer) | Yes | 100% Core MATLAB |
| **Spatial Interpolation (`scatteredInterpolant`)** | Base MATLAB | Yes | Built-in |
| **Physics-Informed GPR** | None (Native Vectorized Covariance) | Yes | Built-in |
| **Random Forest Ensemble** | Statistics & Machine Learning (Optional) | Yes (150-Tree Native Engine) | Built-in |
| **Theoretical Propagation Reference** | Communications Toolbox (Optional) | Yes (3GPP / FSPL Formulas) | Built-in |
| **Active Learning & Visualization** | Base MATLAB | Yes | Built-in |

---

## 🛠️ Design for Portability

To ensure universal reproducibility on any MATLAB environment (student laptops, grading servers, university clusters):
- All advanced algorithms (Gaussian Process Regression with Matérn 5/2 kernel, Cholesky factorization, Random Forest ensemble, $k$-NN IDW, and 3GPP propagation baselines) are implemented in **high-performance, vectorized native MATLAB**.
- If optional toolboxes (`Statistics and Machine Learning Toolbox` or `Communications Toolbox`) are installed, the codebase automatically detects and leverages them. If they are absent, the native pure MATLAB engines execute seamlessly with zero degradation in accuracy.

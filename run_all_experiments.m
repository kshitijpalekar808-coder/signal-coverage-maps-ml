%% MASTER RUNNER: ALL EXPERIMENTAL PROTOCOLS (MathWorks Challenge #151)
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Description:
% Executes the complete end-to-end scientific and experimental pipeline:
%   1. Data generation, validation, and zero-leakage route partitioning
%   2. Primary Model Benchmark on held-out unseen corridors (Routes 10-12)
%   3. Simplified 3GPP-inspired propagation model comparison (reference only)
%   4. Scientific Feature Ablation Study (Spatial vs. Physics vs. Hybrid)
%   5. Controlled Synthetic Ground-Truth Sparsity Sweep (2% to 40%)
%   6. Route-Aware Active Learning Drive-Test Optimization (20 Trials)
%   7. Publication-Quality Coverage, Uncertainty, and Error Residual Mapping
% =========================================================================

clear; clc; close all;

projectRoot = fileparts(mfilename('fullpath'));
if isempty(projectRoot), projectRoot = pwd; end
addpath(genpath(fullfile(projectRoot, 'src')));

if ~exist(fullfile(projectRoot, 'data'), 'dir'), mkdir(fullfile(projectRoot, 'data')); end
if ~exist(fullfile(projectRoot, 'results'), 'dir'), mkdir(fullfile(projectRoot, 'results')); end
if ~exist(fullfile(projectRoot, 'results', 'figures'), 'dir'), mkdir(fullfile(projectRoot, 'results', 'figures')); end

fprintf('======================================================================\n');
fprintf('  SIGNAL COVERAGE MAPPING USING MEASUREMENTS & ML (MathWorks #151)\n');
fprintf('======================================================================\n\n');

%% =========================================================================
%% 0. REAL-WORLD FIELD DATA BENCHMARK (mySignals GSM Dataset)
%% =========================================================================
fprintf('>>> STEP 0: Running Authentic Field Telemetry Benchmark (mySignals GSM-1800)...\n');
realFieldTable = run_real_field_benchmark();

%% =========================================================================
%% 1. LOAD SYNTHETIC GRID DATASET (Controlled Ground-Truth Experiments)
%% =========================================================================
fprintf('\n>>> STEP 1: Loading Synthetic Urban Grid Dataset & Applying Route-Based Spatial Holdout...\n');
csvPath = fullfile(projectRoot, 'data', 'synthetic_urban_grid_dataset.csv');
testRoutes = [10, 11, 12];
[trainData, testData, fullGrid, bsInfo, allData] = load_urban_drive_test_data(csvPath, 'route_holdout', testRoutes);

%% =========================================================================
%% 2. TRAIN & EVALUATE PRIMARY MODEL BENCHMARK ON UNSEEN ROUTES
%% =========================================================================
fprintf('\n>>> STEP 2: Training & Benchmarking Models on Unseen Held-Out Routes...\n');

% 1. Theoretical 3GPP Propagation (No training)
theoModel = theoretical_propagation_model(bsInfo);
pred_theo = theoModel.predict(testData);
m_theo = evaluate_predictions(testData.RSRP, pred_theo);

% 2. Empirical Log-Distance Path Loss (OLS)
logModel = log_distance_model(trainData);
pred_log = logModel.predict(testData);
m_log = evaluate_predictions(testData.RSRP, pred_log);

% 3. True Mathematical kNN-IDW (p=2, k=30)
idwModel = knn_idw_model(trainData, 2, 30);
pred_idw = idwModel.predict(testData);
m_idw = evaluate_predictions(testData.RSRP, pred_idw);

% 4. Natural Neighbor Spatial Interpolation
natModel = natural_neighbor_model(trainData);
pred_nat = natModel.predict(testData);
m_nat = evaluate_predictions(testData.RSRP, pred_nat);

% 5. Random Forest Regression (150 Trees)
rfModel = random_forest_model(trainData, {'X', 'Y', 'Distance', 'LogDist', 'Azimuth'}, 150, 42);
pred_rf = rfModel.predict(testData);
m_rf = evaluate_predictions(testData.RSRP, pred_rf);

% 6. Proposed: Spatial ARD GPR [X, Y] (Hyperparams optimized on train only)
gprModel = physics_informed_gpr_model(trainData, {'X', 'Y'});
[pred_gpr_sp, sigma_latent_te, ~] = gprModel.predict(testData);
m_gpr_sp = evaluate_predictions(testData.RSRP, pred_gpr_sp);

% 7. Hybrid Physics-Informed ARD GPR [All 5 Features] (Collinear Comparison)
gprHybrid = physics_informed_gpr_model(trainData, {'X', 'Y', 'Distance', 'LogDist', 'Azimuth'});
pred_gpr_hy = gprHybrid.predict(testData);
m_gpr_hy = evaluate_predictions(testData.RSRP, pred_gpr_hy);

% Compile primary benchmark table
modelNames = {
    '1. 3GPP-Inspired UMi Reference (No Training)'; ...
    '2. Empirical Log-Distance Fit (OLS)'; ...
    '3. True kNN-IDW (p=2, k=30)'; ...
    '4. Natural Neighbor Interpolation'; ...
    '5. Random Forest (150 Trees)'; ...
    '6. Proposed: Spatial ARD GPR [X, Y]'; ...
    '7. Hybrid ARD GPR [All 5 Features]'
};

rmse_vals = [m_theo.RMSE; m_log.RMSE; m_idw.RMSE; m_nat.RMSE; m_rf.RMSE; m_gpr_sp.RMSE; m_gpr_hy.RMSE];
mae_vals  = [m_theo.MAE;  m_log.MAE;  m_idw.MAE;  m_nat.MAE;  m_rf.MAE;  m_gpr_sp.MAE;  m_gpr_hy.MAE];
r2_vals   = [m_theo.R2;   m_log.R2;   m_idw.R2;   m_nat.R2;   m_rf.R2;   m_gpr_sp.R2;   m_gpr_hy.R2];
maxae_vals= [m_theo.MaxAE;m_log.MaxAE;m_idw.MaxAE;m_nat.MaxAE;m_rf.MaxAE;m_gpr_sp.MaxAE;m_gpr_hy.MaxAE];
p95ae_vals= [m_theo.P95AE;m_log.P95AE;m_idw.P95AE;m_nat.P95AE;m_rf.P95AE;m_gpr_sp.P95AE;m_gpr_hy.P95AE];

benchmarkTable = table(modelNames, rmse_vals, mae_vals, r2_vals, maxae_vals, p95ae_vals, ...
    'VariableNames', {'Model', 'RMSE_dB', 'MAE_dB', 'R2_Score', 'MaxAE_dB', 'P95AE_dB'});

fprintf('\n=== PRIMARY MODEL BENCHMARK (HELD-OUT ROUTES 10-12) ===\n');
disp(benchmarkTable);

writetable(benchmarkTable, fullfile(projectRoot, 'results', 'benchmark_results.csv'));
writetable(benchmarkTable, fullfile(projectRoot, 'results', 'real_data_benchmark_results.csv')); % Backward compatibility

%% =========================================================================
%% 3. SCIENTIFIC FEATURE ABLATION STUDY
%% =========================================================================
fprintf('\n>>> STEP 3: Running Scientific Feature Ablation Study...\n');

ablationConfigs = {
    {'X', 'Y'}, ...
    {'Distance', 'LogDist', 'Azimuth'}, ...
    {'X', 'Y', 'Distance', 'LogDist', 'Azimuth'}
};
configNames = {
    'Config A: Spatial Only [X, Y]'; ...
    'Config B: Radio Physics Only [Distance, LogDist, Azimuth]'; ...
    'Config C: Full Hybrid [All 5 Features]'
};

numConfigs = length(ablationConfigs);
gpr_abl_rmse = zeros(numConfigs, 1);
gpr_abl_r2   = zeros(numConfigs, 1);
rf_abl_rmse  = zeros(numConfigs, 1);
rf_abl_r2    = zeros(numConfigs, 1);

for i = 1:numConfigs
    feats = ablationConfigs{i};
    % Train ARD GPR — hyperparameters optimized on training data only
    % NOTE: For fair ablation, each feature subset gets its own ARD optimization
    % (no shared hyperparameters across configurations).
    gpr_abl = physics_informed_gpr_model(trainData, feats);
    [p_gpr, ~, ~] = gpr_abl.predict(testData);
    m_gpr_abl = evaluate_predictions(testData.RSRP, p_gpr);
    gpr_abl_rmse(i) = m_gpr_abl.RMSE;
    gpr_abl_r2(i)   = m_gpr_abl.R2;

    % Train Random Forest (150 Trees)
    rf_abl = random_forest_model(trainData, feats, 150, 42);
    p_rf = rf_abl.predict(testData);
    m_rf_abl = evaluate_predictions(testData.RSRP, p_rf);
    rf_abl_rmse(i)  = m_rf_abl.RMSE;
    rf_abl_r2(i)    = m_rf_abl.R2;
end

ablationTable = table(configNames, gpr_abl_rmse, gpr_abl_r2, rf_abl_rmse, rf_abl_r2, ...
    'VariableNames', {'FeatureConfiguration', 'GPR_RMSE_dB', 'GPR_R2', 'RF_RMSE_dB', 'RF_R2'});

fprintf('\n=== FEATURE ABLATION RESULTS ===\n');
disp(ablationTable);

writetable(ablationTable, fullfile(projectRoot, 'results', 'ablation_study_results.csv'));
figAblation = plot_ablation_comparison(ablationTable, true);

%% =========================================================================
%% 4. CONTROLLED SYNTHETIC GROUND-TRUTH SPARSITY SWEEP
%% =========================================================================
fprintf('\n>>> STEP 4: Running Controlled Synthetic Sparsity Stress Test...\n');

sparsities = [0.02, 0.05, 0.08, 0.10, 0.20, 0.40];
[sparsityTable, figSparsity] = run_sparsity_analysis(sparsities, true);

%% =========================================================================
%% 5. ROUTE-AWARE ACTIVE LEARNING OPTIMIZATION (20 REPEATED TRIALS)
%% =========================================================================
fprintf('\n>>> STEP 5: Running Route-Aware Active Learning Optimization...\n');
[alTable, figAL] = run_route_active_learning(20, true);

%% =========================================================================
%% 6. PUBLICATION-QUALITY COVERAGE & UNCERTAINTY VISUALIZATION
%% =========================================================================
fprintf('\n>>> STEP 6: Generating Continuous 2D Coverage & Uncertainty Maps...\n');
figCoverage = plot_coverage_maps(fullGrid, trainData, testData, gprModel, bsInfo, true);

%% =========================================================================
%% 7. PREDICTOR FEATURE IMPORTANCE
%% =========================================================================
figImportance = figure('Name', 'Predictor Importance', 'Color', 'w', 'Position', [150, 150, 750, 480]);
importances = rfModel.OOBPermutedPredictorImportance;
featureLabels = {'X (East)', 'Y (North)', 'Distance (m)', 'LogDist', 'Azimuth (\theta)'};

barh(importances, 'FaceColor', [0.2 0.6 0.85]);
set(gca, 'YTickLabel', featureLabels, 'FontSize', 11, 'XColor', 'k', 'YColor', 'k');
xlabel('Out-of-Bag (OOB) Mean Squared Error Increase (dB^2)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
title('Random Forest Predictor Feature Importance', 'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');
grid on; box on;
saveas(figImportance, fullfile(projectRoot, 'results', 'figures', 'predictor_importance.png'));
saveas(figImportance, fullfile(projectRoot, 'results', 'real_data_feature_importance.png'));

%% =========================================================================
%% 8. FINAL SUMMARY & INTEGRITY CHECK
%% =========================================================================
fprintf('\n======================================================================\n');
fprintf('  ALL EXPERIMENTS COMPLETED SUCCESSFULLY!\n');
fprintf('  - All results exported to "results/" directory.\n');
fprintf('  - All figures exported to "results/figures/" directory.\n');
fprintf('======================================================================\n\n');

validate_project();

%% QUICK DEMO: SIGNAL COVERAGE MAPPING INFERENCE (MathWorks Challenge #151)
% =========================================================================
% Rapid 5-Second Verification Script
% Loads pre-trained model from models/ and evaluates on data/sample/
% Fulfills MathWorks Repository Guideline 6 (Fast Test/Demo)
% =========================================================================

clear; clc; close all;
tic;

fprintf('======================================================================\n');
fprintf('  QUICK DEMO: PRE-TRAINED SIGNAL COVERAGE PREDICTION\n');
fprintf('  MathWorks Excellence in Innovation - Project #151\n');
fprintf('======================================================================\n\n');

projectRoot = fileparts(mfilename('fullpath'));
if isempty(projectRoot), projectRoot = pwd; end
addpath(genpath(fullfile(projectRoot, 'src')));

%% 1. Load Pre-Trained Model
modelPath = fullfile(projectRoot, 'models', 'gpr_spatial_model.mat');
if ~exist(modelPath, 'file')
    error('Pre-trained model not found at %s. Run setup_submission_assets first.', modelPath);
end
fprintf('1. Loading pre-trained Spatial ARD GPR model from models/...\n');
modelArtifact = load(modelPath);
gprModel = modelArtifact.gprModel;

%% 2. Load Sample Data & Project Coordinates
samplePath = fullfile(projectRoot, 'data', 'sample', 'sample_urban_grid.csv');
if ~exist(samplePath, 'file')
    error('Sample data not found at %s.', samplePath);
end
fprintf('2. Loading sample drive-test measurements from data/sample/...\n');
[~, ~, ~, ~, processedSample] = load_urban_drive_test_data(samplePath, 'none');

%% 4. Execute Inference with Analytical Uncertainty
fprintf('3. Executing GPR inference on %d sample measurement points...\n', height(processedSample));
[predRSRP, sigmaLatent, yCI] = gprModel.predict(processedSample);

%% 5. Evaluate Metrics
metrics = evaluate_predictions(processedSample.RSRP, predRSRP);

fprintf('\n----------------- VERIFICATION METRICS -----------------\n');
fprintf('  Evaluated Points : %d\n', height(processedSample));
fprintf('  RMSE             : %.2f dB\n', metrics.RMSE);
fprintf('  MAE              : %.2f dB\n', metrics.MAE);
fprintf('  R^2 Score        : %.4f\n', metrics.R2);
fprintf('  Max Error        : %.2f dB\n', metrics.MaxAE);
fprintf('  Mean Latent Std  : %.2f dB (GPR Bayesian Uncertainty)\n', mean(sigmaLatent));
fprintf('--------------------------------------------------------\n\n');

%% 6. Generate Quick Verification Figure
fig = figure('Name', 'Quick Demo Verification', 'Color', 'w', 'Position', [150, 150, 850, 380]);

subplot(1, 2, 1);
scatter(processedSample.X, processedSample.Y, 35, processedSample.RSRP, 'filled');
colorbar; colormap(gca, turbo);
title(sprintf('Sample Drive-Test RSRP (N=%d)', height(processedSample)), 'FontSize', 11);
xlabel('Local X [m]'); ylabel('Local Y [m]'); grid on; axis equal;

subplot(1, 2, 2);
scatter(processedSample.RSRP, predRSRP, 30, sigmaLatent, 'filled');
hold on;
minVal = min([processedSample.RSRP; predRSRP]) - 2;
maxVal = max([processedSample.RSRP; predRSRP]) + 2;
plot([minVal, maxVal], [minVal, maxVal], 'k--', 'LineWidth', 1.5);
cb = colorbar; cb.Label.String = 'Latent \sigma (dB)';
colormap(gca, parula);
title(sprintf('Prediction Correlation (R^2 = %.3f)', metrics.R2), 'FontSize', 11);
xlabel('Measured RSRP [dBm]'); ylabel('Predicted RSRP [dBm]'); grid on; axis equal;

elapsedTime = toc;
fprintf('>>> [SUCCESS] Quick demo executed cleanly in %.2f seconds!\n\n', elapsedTime);

%% =========================================================================
% SCRIPT: Run Feature Ablation Study
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Description:
% Quantifies the predictive value of domain-specific radio features by
% comparing:
%   - Config A: Spatial Coordinates Only [X, Y]
%   - Config B: Radio Physics Only [Distance, LogDist, Azimuth]
%   - Config C: Full Hybrid Physics-Informed Model [All 5 Features]
% Across both GPR and Random Forest regressors on held-out test routes.
% =========================================================================

clear; clc; close all;

projectRoot = fileparts(mfilename('fullpath'));
if isempty(projectRoot), projectRoot = pwd; end
addpath(genpath(fullfile(projectRoot, 'src')));

fprintf('======================================================================\n');
fprintf('  RUNNING FEATURE ABLATION STUDY (Physics vs Spatial Coordinates)\n');
fprintf('======================================================================\n\n');

% Load dataset with route holdout (Train: Routes 1-9, Test: Routes 10-12)
[trainData, testData] = load_urban_drive_test_data([], 'route_holdout', [10, 11, 12]);

% Run ablation study
[ablationTable, fig] = evaluate_feature_ablation(trainData, testData, true);

disp('=== Feature Ablation Benchmark Results ===');
disp(ablationTable);

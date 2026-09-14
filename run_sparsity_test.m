%% =========================================================================
% SCRIPT: Run Sparsity Stress Test
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Description:
% Evaluates reconstruction accuracy (RMSE) across measurement densities
% [2%, 5%, 8%, 10%, 20%, 40%] on controlled continuous 2D ground truth
% (N = 10,201 points) to benchmark kNN-IDW, Log-Distance, Random Forest,
% and Physics-Informed GPR under extreme measurement sparsity.
% =========================================================================

clear; clc; close all;

projectRoot = fileparts(mfilename('fullpath'));
if isempty(projectRoot), projectRoot = pwd; end
addpath(genpath(fullfile(projectRoot, 'src')));

fprintf('======================================================================\n');
fprintf('  RUNNING CONTROLLED CONTINUOUS SPARSITY STRESS TEST (2%% to 40%%)\n');
fprintf('======================================================================\n\n');

% Measurement density levels (2% to 40%)
sparsities = [0.02, 0.05, 0.08, 0.10, 0.20, 0.40];

[sparsityTable, fig] = run_sparsity_analysis(sparsities, true);

writetable(sparsityTable, fullfile(projectRoot, 'results', 'sparsity_stress_results.csv'));
writetable(sparsityTable, fullfile(projectRoot, 'results', 'sparsity_results.csv'));

disp('=== Sparsity Stress Test Results ===');
disp(sparsityTable);

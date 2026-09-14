%% =========================================================================
% SCRIPT: Run Route-Aware Active Learning & Drive-Test Optimization
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Description:
% Evaluates Route-Aware Active Learning:
% Tests whether adaptively selecting the next street route with highest
% GPR latent prediction uncertainty sigma_latent(x,y) improves generalization
% on completely unseen held-out routes (Routes 10-12) compared to passive
% random route selection.
% Runs over 20 repeated trials and reports Mean +/- Std.
% =========================================================================

clear; clc; close all;

projectRoot = fileparts(mfilename('fullpath'));
if isempty(projectRoot), projectRoot = pwd; end
addpath(genpath(fullfile(projectRoot, 'src')));

fprintf('======================================================================\n');
fprintf('  RUNNING ROUTE-AWARE ACTIVE LEARNING DRIVE-TEST OPTIMIZATION\n');
fprintf('======================================================================\n\n');

numTrials = 20;
[alTable, fig] = run_route_active_learning(numTrials, true);

disp('=== Active Learning Optimization Results (Mean +/- Std) ===');
disp(alTable);

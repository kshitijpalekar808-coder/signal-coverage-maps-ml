%% =========================================================================
% MASTER SCRIPT: Signal Coverage Maps Using Measurements & Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Reference: https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/tree/main/projects/Signal%20Coverage%20Maps%20Using%20Measurements%20and%20Machine%20Learning
%
% Quick Start:
%   main                    <- Runs complete benchmark, ablation, active learning & mapping
%   run_all_experiments    <- Executes all experimental protocols and exports figures/CSVs
%   validate_project        <- Runs automated scientific & zero-leakage integrity checks
% =========================================================================

clear; clc; close all;

fprintf('======================================================================\n');
fprintf('  STARTING MATHWORKS PROJECT #151 BENCHMARK PIPELINE\n');
fprintf('======================================================================\n\n');

% Execute complete experimental suite
run_all_experiments;

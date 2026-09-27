%% =========================================================================
% MASTER SCRIPT: Signal Coverage Maps Using Measurements & Machine Learning
% MathWorks Excellence in Innovation - Project #151
% Author: Kshitij Palekar (kshitij.palekar@vit.edu.in)
% =========================================================================
% Reference: https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/tree/main/projects/Signal%20Coverage%20Maps%20Using%20Measurements%20and%20Machine%20Learning
%
% Available Execution Options:
%   quick_demo              <- Rapid 5-second verification using pre-trained model & sample data
%   main                    <- Runs complete benchmark, ablation, active learning & mapping
%   validate_project        <- Runs automated 12-stage scientific & zero-leakage verification suite
%   runtests('tests')       <- Executes official MATLAB Unit Test suite (tests/test_signal_coverage_pipeline.m)
% =========================================================================

clear; clc; close all;

fprintf('======================================================================\n');
fprintf('  STARTING MATHWORKS PROJECT #151 BENCHMARK PIPELINE\n');
fprintf('  Signal Coverage Maps Using Measurements & Machine Learning\n');
fprintf('======================================================================\n\n');

% Execute complete experimental suite end-to-end
run_all_experiments;

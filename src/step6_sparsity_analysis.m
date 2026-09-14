%% BACKWARD-COMPATIBILITY ADAPTER: STEP 6 SPARSITY ANALYSIS
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Bridges legacy calls to step6_sparsity_analysis(...) to run_sparsity_analysis.
% =========================================================================

function [sparsityTable, fig] = step6_sparsity_analysis(sparsities, saveFig)
    if nargin < 1 || isempty(sparsities)
        sparsities = [0.02, 0.05, 0.08, 0.10, 0.20, 0.40];
    end
    if nargin < 2 || isempty(saveFig)
        saveFig = true;
    end
    [sparsityTable, fig] = run_sparsity_analysis(sparsities, saveFig);
end

%% BACKWARD-COMPATIBILITY ADAPTER: STEP 7 ABLATION STUDY
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Bridges legacy calls to step7_ablation_study(...) to run_ablation_study.
% =========================================================================

function [ablationTable, fig] = step7_ablation_study(trainData, testData, saveFig)
    if nargin < 3 || isempty(saveFig)
        saveFig = true;
    end
    if nargin < 1, trainData = []; end
    if nargin < 2, testData = []; end
    [ablationTable, fig] = evaluate_feature_ablation(trainData, testData, saveFig);
end

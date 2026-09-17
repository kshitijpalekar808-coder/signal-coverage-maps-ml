%% BACKWARD-COMPATIBILITY ADAPTER: STEP 1 PREPARE DATA
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Bridges legacy calls to step1_prepare_data(...) to generate_sparsity_dataset.
% =========================================================================

function [trainData, testData, fullGrid, groundTruth, X_mat, Y_mat] = step1_prepare_data(sparsityRatio, saveData)
    if nargin < 1 || isempty(sparsityRatio)
        sparsityRatio = 0.08;
    end
    [trainData, testData, fullGrid, groundTruth, X_mat, Y_mat] = generate_sparsity_dataset(sparsityRatio);
end

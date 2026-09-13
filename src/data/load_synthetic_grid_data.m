%% DATA LOADER: SYNTHETIC URBAN GRID DATASET
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Canonical data loader for controlled synthetic Manhattan-grid drive tests.
% Evaluates across 12 explicit street routes (Routes 1-9 train, 10-12 test).
% =========================================================================

function [trainData, testData, fullGrid, bsInfo, allData] = load_synthetic_grid_data(csvFilePath, splitMode, testRoutes)
    if nargin < 1, csvFilePath = []; end
    if nargin < 2, splitMode = []; end
    if nargin < 3, testRoutes = []; end
    [trainData, testData, fullGrid, bsInfo, allData] = load_urban_drive_test_data(csvFilePath, splitMode, testRoutes);
end

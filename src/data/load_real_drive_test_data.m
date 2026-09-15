%% BACKWARD-COMPATIBILITY ADAPTER: LOAD REAL DRIVE TEST DATA
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Bridges legacy calls to load_real_drive_test_data(...) to load_urban_drive_test_data.
% =========================================================================

function [trainData, testData, fullGrid, bsInfo, allData] = load_real_drive_test_data(varargin)
    [trainData, testData, fullGrid, bsInfo, allData] = load_urban_drive_test_data(varargin{:});
end

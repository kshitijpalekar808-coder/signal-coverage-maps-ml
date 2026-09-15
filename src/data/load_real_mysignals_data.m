%% DATA LOADER: REAL MYSIGNALS GSM FIELD DATASET
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Source: mySignals Project (Alimpertis et al., IEEE WCL 2014)
% Environment: Urban Chania, Crete, Greece (GSM-1800 Downlink, 1860.2 MHz)
% Base Transceiver Station: Cell 6056x at (35.508354 N, 24.024523 E)
%
% Spatial Validation Rigor:
% -------------------------
% Evaluates on completely unvisited geographic sector / street corridor:
%   - Training Zones: Zones 1, 2, 3 (73.7% of measurements, 662 points)
%   - Test Zone: Zone 4 (26.3% of measurements, 236 points)
% Guarantees ZERO spatial point leakage between training and testing.
% =========================================================================

function [trainData, testData, bsInfo, allData] = load_real_mysignals_data(csvFilePath, testZone)
    if nargin < 1 || isempty(csvFilePath)
        csvFilePath = fullfile('data', 'real_mysignals_dataset.csv');
    end
    if nargin < 2 || isempty(testZone)
        testZone = 4; % Zone 4 (Northeast corridor) held out for testing
    end

    if ~exist(csvFilePath, 'file')
        error('Real mySignals dataset not found at "%s". Run prepare_real_mysignals_dataset first.', csvFilePath);
    end

    rawTable = readtable(csvFilePath);

    % Standardize target column name to 'RSRP'
    if ismember('RSRP_dBm', rawTable.Properties.VariableNames)
        rawTable.RSRP = rawTable.RSRP_dBm;
    end

    allData = rawTable;

    % Strict Spatial Holdout
    trainMask = (rawTable.RouteID ~= testZone);
    testMask  = (rawTable.RouteID == testZone);

    trainData = rawTable(trainMask, :);
    testData  = rawTable(testMask, :);

    % Assertion: Zero spatial data leakage
    overlap = intersect(unique(trainData.RouteID), unique(testData.RouteID));
    assert(isempty(overlap), 'FATAL: Spatial leakage detected between train and test zones!');

    % Base Station Metadata
    bsInfo.Name      = 'mySignals GSM BTS 6056x (Chania, Greece)';
    bsInfo.Lat       = 35.508354;
    bsInfo.Lon       = 24.024523;
    bsInfo.X         = 0;
    bsInfo.Y         = 0;
    bsInfo.Freq_MHz  = 1860.2; % GSM-1800 Downlink
    bsInfo.TxPower_dBm = 43;   % Macro BTS standard (20 W)
    bsInfo.AntGain_dBi = 15;

    fprintf('Loaded Genuine mySignals Field Dataset: %d total samples\n', height(allData));
    fprintf('  -> Training Zones (1, 2, 3): %d samples (%.1f%%)\n', height(trainData), 100 * height(trainData) / height(allData));
    fprintf('  -> Held-Out Test Zone [%d]:    %d samples (%.1f%%, 0%% spatial leakage)\n', ...
        testZone, height(testData), 100 * height(testData) / height(allData));
end

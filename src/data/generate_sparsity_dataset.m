%% DATA GENERATOR: CONTROLLED CONTINUOUS SYNTHETIC GRID SPARSITY DATASET
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Description:
% Generates continuous 2D ground-truth signal coverage over a 1000m x 1000m
% urban area (10m resolution = 10,201 points) with known base station at
% (500m, 500m), 3GPP-inspired path loss, and correlated urban shadow fading.
% Samples sparse training probes at a specified sampling ratio s (e.g. 0.02,
% 0.05, 0.08, 0.10, 0.20, 0.40) with 1.5 dB observation noise.
% The remaining (1 - s) grid points form the test evaluation set.
% =========================================================================

function [trainData, testData, fullGrid, groundTruth, X_mat, Y_mat] = generate_sparsity_dataset(sparsityRatio, randomSeed)
    if nargin < 1 || isempty(sparsityRatio)
        sparsityRatio = 0.08; % Default: 8% sparse sampling
    end
    if nargin < 2 || isempty(randomSeed)
        randomSeed = 42;
    end

    rng(randomSeed, 'twister');

    % Grid Specifications (1000m x 1000m at 10m resolution = 10,201 nodes)
    gridSize = 1000;
    res = 10;
    [X_mat, Y_mat] = meshgrid(0:res:gridSize, 0:res:gridSize);
    totalNodes = numel(X_mat);

    % Base Station Parameters
    tx_x = 500;
    tx_y = 500;
    tx_power_dBm = 43;      % 20 Watts (43 dBm)
    antenna_gain_dBi = 15;  % 15 dBi sector antenna
    freq_GHz = 2.1;         % 2.1 GHz carrier

    % Feature Extraction: Distance, Log-Distance, Azimuth
    dist_mat = sqrt((X_mat - tx_x).^2 + (Y_mat - tx_y).^2) + 1; % 1m offset avoids log(0)
    logDist_mat = log10(dist_mat);
    azimuth_mat = atan2d(Y_mat - tx_y, X_mat - tx_x);

    % Physical Propagation: 3GPP-inspired path loss + correlated urban shadowing
    pathLoss_mat = 32.4 + 20 * log10(freq_GHz) + 30 * log10(dist_mat);
    shadowing_mat = 9.5 * sin(X_mat / 75) .* cos(Y_mat / 75) + 3.8 * cos(dist_mat / 45);

    % Ground Truth RSRP (dBm)
    groundTruth = tx_power_dBm + antenna_gain_dBi - pathLoss_mat + shadowing_mat;

    % Flat vectors for table construction
    X_vec = X_mat(:);
    Y_vec = Y_mat(:);
    dist_vec = dist_mat(:);
    logDist_vec = logDist_mat(:);
    azimuth_vec = azimuth_mat(:);
    rsrp_vec = groundTruth(:);

    fullGrid = table(X_vec, Y_vec, dist_vec, logDist_vec, azimuth_vec, rsrp_vec, ...
        'VariableNames', {'X', 'Y', 'Distance', 'LogDist', 'Azimuth', 'RSRP'});

    % Sample training indices at sparsityRatio
    numTrain = max(10, round(totalNodes * sparsityRatio));
    permIndices = randperm(totalNodes);
    trainIndices = permIndices(1:numTrain);
    testIndices  = permIndices(numTrain + 1:end);

    % Training set with 1.5 dB observation noise (Gaussian fast-fading & measurement noise)
    noise_sigma = 1.5;
    train_noise = noise_sigma * randn(numTrain, 1);
    trainRSRP = rsrp_vec(trainIndices) + train_noise;

    trainData = table(X_vec(trainIndices), Y_vec(trainIndices), dist_vec(trainIndices), ...
                      logDist_vec(trainIndices), azimuth_vec(trainIndices), trainRSRP, ...
                      'VariableNames', {'X', 'Y', 'Distance', 'LogDist', 'Azimuth', 'RSRP'});

    % Test set: remaining ground truth nodes (zero noise, evaluating continuous field reconstruction)
    testData = table(X_vec(testIndices), Y_vec(testIndices), dist_vec(testIndices), ...
                     logDist_vec(testIndices), azimuth_vec(testIndices), rsrp_vec(testIndices), ...
                     'VariableNames', {'X', 'Y', 'Distance', 'LogDist', 'Azimuth', 'RSRP'});
end

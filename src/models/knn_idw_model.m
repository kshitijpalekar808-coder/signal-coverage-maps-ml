% k-Nearest Neighbor Inverse Distance Weighting (kNN-IDW) Model
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% Author: Kshitij Palekar
%
% Formulation:
%   w_i = 1 / (d_i^p)
%   y_hat(x) = sum_{i=1}^k (w_i * y_i) / sum_{i=1}^k (w_i)
%
% Default Parameters:
%   p = 2 (inverse-square distance weighting)
%   k = 30 nearest neighbors

function idwModel = knn_idw_model(trainData, p, k)
    if nargin < 2 || isempty(p), p = 2; end
    if nargin < 3 || isempty(k), k = 30; end

    trainX = trainData.X;
    trainY = trainData.Y;
    trainRSRP = trainData.RSRP;

    idwModel.trainX = trainX;
    idwModel.trainY = trainY;
    idwModel.trainRSRP = trainRSRP;
    idwModel.p = p;
    idwModel.k = min(k, length(trainX));
    idwModel.name = sprintf('kNN-IDW (p=%d, k=%d)', p, idwModel.k);

    idwModel.predict = @(evalData) predictKNN_IDW(evalData, trainX, trainY, trainRSRP, p, idwModel.k);
end

function preds = predictKNN_IDW(evalData, trainX, trainY, trainRSRP, p, k)
    if istable(evalData)
        evalX = evalData.X;
        evalY = evalData.Y;
    elseif ismatrix(evalData) && size(evalData, 2) >= 2
        evalX = evalData(:, 1);
        evalY = evalData(:, 2);
    else
        error('Invalid evaluation data format for IDW.');
    end

    M = length(evalX);
    N = length(trainX);
    k = min(k, N);
    preds = zeros(M, 1);

    % Memory-efficient chunked evaluation
    chunkSize = 2000;
    for startIdx = 1:chunkSize:M
        endIdx = min(startIdx + chunkSize - 1, M);
        subX = evalX(startIdx:endIdx);
        subY = evalY(startIdx:endIdx);
        numSub = length(subX);

        % Pairwise 2D Cartesian distance: [numSub x N]
        dx = bsxfun(@minus, subX, trainX');
        dy = bsxfun(@minus, subY, trainY');
        dists = sqrt(dx.^2 + dy.^2);

        % Find k-nearest neighbors for each query point
        [sortedDists, sortIdx] = sort(dists, 2, 'ascend');
        k_dists = sortedDists(:, 1:k);           % [numSub x k]
        k_indices = sortIdx(:, 1:k);             % [numSub x k]
        k_rsrp = trainRSRP(k_indices);           % [numSub x k]

        % Check for exact / coincident points (d < 1e-6)
        isZero = (k_dists(:, 1) < 1e-6);

        % Standard IDW weighting: w_i = 1 / d_i^p
        weights = 1 ./ (k_dists.^p);
        sumWeights = sum(weights, 2);
        weightedSum = sum(weights .* k_rsrp, 2);

        subPreds = weightedSum ./ sumWeights;

        % For exact coincidence, assign the exact measured value directly
        if any(isZero)
            subPreds(isZero) = k_rsrp(isZero, 1);
        end

        preds(startIdx:endIdx) = subPreds;
    end
end

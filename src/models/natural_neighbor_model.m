% Natural Neighbor Spatial Interpolation Model
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% Author: Kshitij Palekar
%
% Delaunay-based natural neighbor interpolation using MATLAB's
% scatteredInterpolant with linear extrapolation fallback.

function natModel = natural_neighbor_model(trainData)
    trainX = trainData.X;
    trainY = trainData.Y;
    trainRSRP = trainData.RSRP;

    interpolant = scatteredInterpolant(trainX, trainY, trainRSRP, 'natural', 'linear');

    natModel.interpolant = interpolant;
    natModel.name = 'Natural Neighbor Spatial Interpolation';
    natModel.defaultFill = mean(trainRSRP);
    natModel.predict = @(evalData) predictNaturalNeighbor(interpolant, evalData, natModel.defaultFill);
end

function preds = predictNaturalNeighbor(interpolant, evalData, defaultFill)
    if istable(evalData)
        evalX = evalData.X;
        evalY = evalData.Y;
    elseif ismatrix(evalData) && size(evalData, 2) >= 2
        evalX = evalData(:, 1);
        evalY = evalData(:, 2);
    else
        error('Invalid evaluation data format for Natural Neighbor.');
    end

    preds = interpolant(evalX, evalY);

    % Handle potential NaN outputs outside convex hull
    if any(isnan(preds))
        preds(isnan(preds)) = defaultFill;
    end
end

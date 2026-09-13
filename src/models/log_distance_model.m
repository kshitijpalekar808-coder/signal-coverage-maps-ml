% Empirical Log-Distance Path Loss Model (OLS Regression)
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% Author: Kshitij Palekar
%
% Ordinary Least Squares fit:
%   RSRP(d) = beta_0 + beta_1 * log10(d)
% where beta_0 is reference intercept power (dBm) and beta_1 is -10*n.

function logModel = log_distance_model(trainData)
    X_mat = [ones(height(trainData), 1), trainData.LogDist];
    y_vec = trainData.RSRP;

    % OLS Solution: beta = (X^T X)^-1 X^T y
    b = X_mat \ y_vec;

    logModel.beta = b;
    logModel.intercept = b(1);
    logModel.slope = b(2);
    logModel.pathLossExponent = -b(2) / 10;
    logModel.name = 'Empirical Log-Distance Model (OLS)';
    logModel.predict = @(inputData) predictLogDistance(inputData, b);

    fprintf('Fitted Empirical Log-Distance Model: RSRP = %.2f + (%.2f) * log10(d) [PL Exponent n = %.2f]\n', ...
        b(1), b(2), logModel.pathLossExponent);
end

function pred_rsrp = predictLogDistance(inputData, b)
    if istable(inputData)
        if ismember('LogDist', inputData.Properties.VariableNames)
            logD = inputData.LogDist;
        elseif ismember('Distance', inputData.Properties.VariableNames)
            logD = log10(max(1, inputData.Distance));
        else
            d = sqrt((inputData.X - 500).^2 + (inputData.Y - 500).^2) + 1;
            logD = log10(d);
        end
    else
        logD = inputData;
    end

    pred_rsrp = [ones(length(logD), 1), logD] * b;
end

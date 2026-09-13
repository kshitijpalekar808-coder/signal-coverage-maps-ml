%% EVALUATION: QUANTITATIVE PREDICTION ERROR BENCHMARKING
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Description:
% Evaluates model predictions against ground truth target measurements on
% held-out spatial test corridors.
%
% Computed Metrics:
%   - RMSE: Root Mean Square Error (dB)
%   - MAE: Mean Absolute Error (dB)
%   - R2: Coefficient of Determination (R^2 Score)
%   - MaxAE: Maximum Absolute Error (dB)
%   - P95AE: 95th Percentile Absolute Error (dB)
% =========================================================================

function metrics = evaluate_predictions(y_actual, y_pred)
    y_actual = y_actual(:);
    y_pred = y_pred(:);

    errors = y_actual - y_pred;
    abs_errors = abs(errors);

    rmse = sqrt(mean(errors.^2));
    mae = mean(abs_errors);
    
    ss_tot = max(eps, sum((y_actual - mean(y_actual)).^2));
    ss_res = sum(errors.^2);
    r2 = 1 - (ss_res / ss_tot);

    max_ae = max(abs_errors);
    p95_ae = prctile(abs_errors, 95);

    metrics.RMSE = rmse;
    metrics.MAE = mae;
    metrics.R2 = r2;
    metrics.MaxAE = max_ae;
    metrics.P95AE = p95_ae;
end

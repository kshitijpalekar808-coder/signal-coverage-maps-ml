%% PREPROCESSING: TRAINING-ONLY FEATURE & TARGET NORMALIZATION
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Function: normalize_features
% Computes feature and target normalization parameters (mean and standard
% deviation) STRICTLY from the training set. Prevents data leakage.
% =========================================================================

function [normParams, X_train_norm, y_train_norm] = normalize_features(X_train_mat, y_train_vec)
    % Compute normalization statistics strictly from training data
    mu_X = mean(X_train_mat, 1);
    std_X = std(X_train_mat, [], 1);
    std_X(std_X == 0) = 1; % Prevent divide-by-zero on constant columns

    if nargin >= 2 && ~isempty(y_train_vec)
        mu_y = mean(y_train_vec);
        std_y = std(y_train_vec);
        if std_y == 0, std_y = 1; end
        y_train_norm = (y_train_vec - mu_y) / std_y;
    else
        mu_y = 0;
        std_y = 1;
        y_train_norm = [];
    end

    X_train_norm = (X_train_mat - mu_X) ./ std_X;

    % Store training parameters
    normParams.mu_X = mu_X;
    normParams.std_X = std_X;
    normParams.mu_y = mu_y;
    normParams.std_y = std_y;

    % Transformation function handle for test sets / evaluation grids
    normParams.transform_X = @(X_mat) (X_mat - mu_X) ./ std_X;
    normParams.transform_y = @(y_vec) (y_vec - mu_y) / std_y;
    normParams.inverse_y   = @(y_norm) (y_norm * std_y) + mu_y;
end

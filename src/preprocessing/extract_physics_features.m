%% PREPROCESSING: PHYSICS FEATURE EXTRACTION
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Function: extract_physics_features
% Computes Euclidean distance d, log10(d), and azimuth theta relative to
% the base station transmitter at (tx_x, tx_y).
% =========================================================================

function featTable = extract_physics_features(X, Y, tx_x, tx_y)
    if nargin < 3 || isempty(tx_x), tx_x = 500; end
    if nargin < 4 || isempty(tx_y), tx_y = 500; end

    dist = sqrt((X - tx_x).^2 + (Y - tx_y).^2) + 1;
    logDist = log10(dist);
    azimuth = atan2d(Y - tx_y, X - tx_x);

    featTable = table(X, Y, dist, logDist, azimuth, ...
        'VariableNames', {'X', 'Y', 'Distance', 'LogDist', 'Azimuth'});
end

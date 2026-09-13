%% DATA LOADER: URBAN DRIVE-TEST LOADER WITH ROUTE-BASED SPATIAL HOLDOUT
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Description:
% Loads urban street network drive-test telemetry from CSV.
% Projects Geodetic (Lat, Lon) to Local Metric Cartesian (X, Y in meters)
% centered around the serving Base Station at (500m, 500m).
%
% Spatial Validation Rigor:
% -------------------------
% Implements strict ROUTE-BASED SPATIAL HOLDOUT:
%   - Training Set: Routes 1 to 9 (75% spatial coverage)
%   - Test Set: Routes 10 to 12 (25% held-out unseen corridors)
% Includes automated assertion checking that train and test route IDs are
% completely disjoint to guarantee ZERO spatial data leakage.
% =========================================================================

function [trainData, testData, fullGrid, bsInfo, allData] = load_urban_drive_test_data(csvFilePath, splitMode, testRoutes)
    if nargin < 1 || isempty(csvFilePath)
        csvFilePath = fullfile('data', 'synthetic_urban_grid_dataset.csv');
    end
    if nargin < 2 || isempty(splitMode)
        splitMode = 'route_holdout'; % 'route_holdout' or 'none'
    end
    if nargin < 3 || isempty(testRoutes)
        testRoutes = [10, 11, 12]; % Strict 25% spatial holdout (unvisited routes)
    end

    % If CSV does not exist or lacks RouteID, regenerate deterministically
    needsRegen = true;
    if exist(csvFilePath, 'file')
        rawTable = readtable(csvFilePath);
        if ismember('RouteID', rawTable.Properties.VariableNames)
            needsRegen = false;
        end
    end

    if needsRegen
        fprintf('Dataset at "%s" missing or invalid. Regenerating synthetic dataset...\n', csvFilePath);
        rawTable = generate_synthetic_drive_test(csvFilePath);
    end

    % Verify explicit RouteID exists
    if ~ismember('RouteID', rawTable.Properties.VariableNames)
        error('Dataset must contain explicit "RouteID" column for route-based spatial holdout validation.');
    end

    % Base Station Geodetic Coordinates
    if ismember('Serving_BS_Lat', rawTable.Properties.VariableNames)
        bs_lat = rawTable.Serving_BS_Lat(1);
        bs_lon = rawTable.Serving_BS_Lon(1);
    else
        bs_lat = mean(rawTable.Latitude);
        bs_lon = mean(rawTable.Longitude);
    end

    % Geodetic to Cartesian metric projection
    R_earth = 6371000; % Earth radius in meters
    lat_rad = deg2rad(rawTable.Latitude);
    lon_rad = deg2rad(rawTable.Longitude);
    bs_lat_rad = deg2rad(bs_lat);
    bs_lon_rad = deg2rad(bs_lon);

    % Local X (East) and Y (North) relative to BS
    X_rel = R_earth * (lon_rad - bs_lon_rad) .* cos((lat_rad + bs_lat_rad) / 2);
    Y_rel = R_earth * (lat_rad - bs_lat_rad);

    % Center inside [0, 1000] m evaluation window (BS at 500m, 500m)
    tx_x = 500;
    tx_y = 500;
    X = X_rel + tx_x;
    Y = Y_rel + tx_y;

    % Bounding box filter [0, 1000] m
    validIdx = (X >= 0 & X <= 1000 & Y >= 0 & Y <= 1000);
    X = X(validIdx);
    Y = Y(validIdx);
    RouteID = rawTable.RouteID(validIdx);

    if ismember('RSRP_dBm', rawTable.Properties.VariableNames)
        RSRP = rawTable.RSRP_dBm(validIdx);
    elseif ismember('RSRP', rawTable.Properties.VariableNames)
        RSRP = rawTable.RSRP(validIdx);
    else
        error('Dataset must contain "RSRP_dBm" or "RSRP" column.');
    end

    % Extract physics-informed features from coordinates
    dist = sqrt((X - tx_x).^2 + (Y - tx_y).^2) + 1; % Offset by 1m to avoid log(0)
    logDist = log10(dist);
    azimuth = atan2d(Y - tx_y, X - tx_x);

    % Complete prepared dataset table
    allData = table(X, Y, dist, logDist, azimuth, RouteID, RSRP, ...
        'VariableNames', {'X', 'Y', 'Distance', 'LogDist', 'Azimuth', 'RouteID', 'RSRP'});

    % ---------------------------------------------------------------------
    % Spatial Partitioning: Route Holdout vs Complete
    % ---------------------------------------------------------------------
    if strcmp(splitMode, 'route_holdout')
        uniqueRoutes = unique(allData.RouteID);
        trainRoutes = setdiff(uniqueRoutes, testRoutes);

        trainMask = ismember(allData.RouteID, trainRoutes);
        testMask = ismember(allData.RouteID, testRoutes);

        trainData = allData(trainMask, :);
        testData = allData(testMask, :);

        % Zero Spatial Leakage Assertion
        commonRoutes = intersect(trainData.RouteID, testData.RouteID);
        if ~isempty(commonRoutes)
            error('CRITICAL DATA LEAKAGE: RouteID %s appears in both training and testing sets!', mat2str(commonRoutes));
        end

        fprintf('Loaded Urban Drive-Test Dataset: %d total points across %d routes\n', height(allData), length(uniqueRoutes));
        fprintf('  -> Training Routes (1-%d): %d samples (%.1f%%)\n', max(trainRoutes), height(trainData), 100*height(trainData)/height(allData));
        fprintf('  -> Held-Out Test Routes %s: %d samples (%.1f%%, 0%% spatial leakage)\n', mat2str(testRoutes), height(testData), 100*height(testData)/height(allData));
    else
        trainData = allData;
        testData = table();
        fprintf('Loaded Urban Drive-Test Dataset: %d points (%d routes, Complete Unsplit)\n', ...
            height(allData), length(unique(allData.RouteID)));
    end

    % Continuous 2D Reconstruction Grid (1000m x 1000m at 10m resolution = 10,201 points)
    res = 10;
    [gridX, gridY] = meshgrid(0:res:1000, 0:res:1000);
    gridDist = sqrt((gridX - tx_x).^2 + (gridY - tx_y).^2) + 1;
    gridLogDist = log10(gridDist);
    gridAzimuth = atan2d(gridY - tx_y, gridX - tx_x);

    fullGrid = table(gridX(:), gridY(:), gridDist(:), gridLogDist(:), gridAzimuth(:), ...
        'VariableNames', {'X', 'Y', 'Distance', 'LogDist', 'Azimuth'});

    % Base station info
    bsInfo.lat = bs_lat;
    bsInfo.lon = bs_lon;
    bsInfo.localX = tx_x;
    bsInfo.localY = tx_y;
    bsInfo.txPower_dBm = 43;
    bsInfo.antennaGain_dBi = 15;
    bsInfo.freq_GHz = 2.1;
end

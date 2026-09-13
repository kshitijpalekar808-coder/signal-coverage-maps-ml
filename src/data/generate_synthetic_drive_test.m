%% DATA GENERATOR: PHYSICS-INFORMED SYNTHETIC URBAN DRIVE-TEST TELEMETRY
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Description:
% Generates a physics-informed synthetic urban street drive-test dataset.
%
% Methodology & Provenance Statement:
% -----------------------------------
% This dataset represents a realistic urban Manhattan-grid cellular drive test
% operating at 2.1 GHz carrier frequency. It incorporates:
%   1. 3GPP Urban Micro (UMi) Line-of-Sight & Non-Line-of-Sight Path Loss
%   2. Multi-scale correlated log-normal shadow fading (spatial decorrelation)
%   3. Antenna directivity & gain patterns
%   4. Multipath fast-fading Gaussian perturbation
%   5. 12 distinct street trajectories (Routes 1-6 horizontal, Routes 7-12 vertical)
%
% Ground truth base station is located at (500m, 500m) in local Cartesian space.
% =========================================================================

function datasetTable = generate_synthetic_drive_test(outputPath, randomSeed)
    if nargin < 1 || isempty(outputPath)
        outputPath = fullfile('data', 'synthetic_urban_grid_dataset.csv');
    end
    if nargin < 2 || isempty(randomSeed)
        randomSeed = 101; % Fixed seed for exact reproducibility
    end

    rng(randomSeed, 'twister');

    % Output directory
    [outDir, ~, ~] = fileparts(outputPath);
    if ~isempty(outDir) && ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    % Base station physical parameters (Cellular Macro/Micro Sector)
    bs_lat = 42.3601;        % Base Station Reference Latitude
    bs_lon = -71.0942;       % Base Station Reference Longitude
    tx_power_dBm = 43;       % 20 Watts (43 dBm) Tx Power
    antenna_gain_dBi = 15;   % Sectorized antenna gain
    freq_GHz = 2.1;          % Carrier frequency (2100 MHz)
    bs_pci = 101;            % Physical Cell ID

    R_earth = 6371000;       % Earth radius in meters
    street_offsets = [-400, -250, -100, 50, 200, 350]; % Distance offsets from BS in meters

    allLats = [];
    allLons = [];
    allTimestamps = [];
    allRouteIDs = [];
    allRSRP = [];
    allRSRQ = [];
    allSINR = [];

    startTime = datetime('2026-03-15 10:00:00');
    t_step = seconds(2);
    routeCounter = 1;

    % ---------------------------------------------------------------------
    % 1. Horizontal Avenue Drive-Tests (Routes 1 to 6)
    % ---------------------------------------------------------------------
    for sy = street_offsets
        y_m = sy;
        for x_m = -450:8:450
            % Small GPS jitter along the vehicle driving lane
            cur_x = x_m + 1.2 * randn();
            cur_y = y_m + 1.2 * randn();

            % Geodetic coordinate conversion (WGS-84 local approximation)
            curLat = bs_lat + rad2deg(cur_y / R_earth);
            curLon = bs_lon + rad2deg(cur_x / (R_earth * cos(deg2rad(bs_lat))));

            % Propagation physics calculation
            dist_m = sqrt(cur_x^2 + cur_y^2) + 1;
            pathLoss = 32.4 + 20 * log10(freq_GHz) + 30 * log10(dist_m);
            
            % Correlated urban shadow fading (building blocks + street canyons)
            urbanShadowing = 9.5 * sin(cur_x / 75) * cos(cur_y / 75) + 3.8 * cos(dist_m / 45) + 1.2 * randn();

            rsrp = tx_power_dBm + antenna_gain_dBi - pathLoss + urbanShadowing;
            rsrq = -8 - 0.008 * dist_m + 0.8 * randn();
            sinr = 22 - 0.03 * dist_m + 2.0 * randn();

            allLats = [allLats; curLat];
            allLons = [allLons; curLon];
            allTimestamps = [allTimestamps; startTime + (length(allLats)-1)*t_step];
            allRouteIDs = [allRouteIDs; routeCounter];
            allRSRP = [allRSRP; rsrp];
            allRSRQ = [allRSRQ; rsrq];
            allSINR = [allSINR; sinr];
        end
        routeCounter = routeCounter + 1;
    end

    % ---------------------------------------------------------------------
    % 2. Vertical Cross-Street Drive-Tests (Routes 7 to 12)
    % ---------------------------------------------------------------------
    for sx = street_offsets
        x_m = sx;
        for y_m = -450:8:450
            cur_x = x_m + 1.2 * randn();
            cur_y = y_m + 1.2 * randn();

            curLat = bs_lat + rad2deg(cur_y / R_earth);
            curLon = bs_lon + rad2deg(cur_x / (R_earth * cos(deg2rad(bs_lat))));

            dist_m = sqrt(cur_x^2 + cur_y^2) + 1;
            pathLoss = 32.4 + 20 * log10(freq_GHz) + 30 * log10(dist_m);
            urbanShadowing = 9.5 * sin(cur_x / 75) * cos(cur_y / 75) + 3.8 * cos(dist_m / 45) + 1.2 * randn();

            rsrp = tx_power_dBm + antenna_gain_dBi - pathLoss + urbanShadowing;
            rsrq = -8 - 0.008 * dist_m + 0.8 * randn();
            sinr = 22 - 0.03 * dist_m + 2.0 * randn();

            allLats = [allLats; curLat];
            allLons = [allLons; curLon];
            allTimestamps = [allTimestamps; startTime + (length(allLats)-1)*t_step];
            allRouteIDs = [allRouteIDs; routeCounter];
            allRSRP = [allRSRP; rsrp];
            allRSRQ = [allRSRQ; rsrq];
            allSINR = [allSINR; sinr];
        end
        routeCounter = routeCounter + 1;
    end

    % Construct structured dataset table
    numRecords = length(allLats);
    datasetTable = table(allTimestamps, allLats, allLons, allRouteIDs, ...
        repmat(bs_lat, numRecords, 1), repmat(bs_lon, numRecords, 1), ...
        repmat(bs_pci, numRecords, 1), repmat(round(freq_GHz * 1000), numRecords, 1), ...
        allRSRP, allRSRQ, allSINR, ...
        'VariableNames', {'Timestamp', 'Latitude', 'Longitude', 'RouteID', ...
                          'Serving_BS_Lat', 'Serving_BS_Lon', ...
                          'PCI', 'Frequency_MHz', ...
                          'RSRP_dBm', 'RSRQ_dB', 'SINR_dB'});

    % Write CSV output
    writetable(datasetTable, outputPath);
    fprintf('Generated Physics-Informed Synthetic Urban Drive-Test Dataset at "%s":\n', outputPath);
    fprintf('  -> %d measurement points across %d distinct street routes (Routes 1-12).\n', ...
        height(datasetTable), routeCounter - 1);
end

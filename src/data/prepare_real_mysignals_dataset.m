%% PREPARE REAL MYSIGNALS DATASET
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Source: mySignals Project (Alimpertis et al., IEEE WCL 2014)
% Base Transceiver Station (BTS): Cell 6056x, Chania, Crete, Greece
% Carrier Frequency: 1860.2 MHz (GSM-1800 Downlink)
% BTS Location: Latitude 35.508354 N, Longitude 24.024523 E
% =========================================================================

function outTable = prepare_real_mysignals_dataset(csvPath, outCsvPath)
    if nargin < 1 || isempty(csvPath)
        csvPath = 'data/mysignals/dataset/measurementsGSM_alldata_cells_6065x/6056x_alldata.csv';
    end
    if nargin < 2 || isempty(outCsvPath)
        outCsvPath = 'data/real_mysignals_dataset.csv';
    end

    if ~exist(csvPath, 'file')
        zipPath = fullfile('data', 'mysignals_dataset.zip');
        if exist(zipPath, 'file')
            fprintf('>>> Extracting raw mySignals dataset archive: %s\n', zipPath);
            unzip(zipPath, fullfile('data', 'mysignals'));
        end
    end

    fprintf('>>> Loading authentic mySignals GSM field dataset from:\n    %s\n', csvPath);

    % Base station location (Chania, Crete)
    bs_lat = 35.508354;
    bs_lon = 24.024523;

    % Read relevant columns
    opts = detectImportOptions(csvPath);
    opts.SelectedVariableNames = {'timestamp', 'rssi', 'latitude', 'longitude', ...
                                  'isMoving', 'cellID', 'freq_dlink', 'horizontalAccuracy'};
    opts.DataLines = [2, 440000];
    raw = readtable(csvPath, opts);

    fprintf('  Read %d raw records. Filtering moving measurements...\n', height(raw));

    % Filter:
    % 1. Moving users (isMoving == 1)
    % 2. Reliable GPS accuracy (<= 65m)
    % 3. Valid cellular RSSI range (-115 dBm to -40 dBm)
    % 4. Non-NaN lat/lon
    valid = (raw.isMoving == 1) & ...
            (raw.horizontalAccuracy <= 65) & ...
            (raw.rssi >= -115) & (raw.rssi <= -40) & ...
            ~isnan(raw.latitude) & ~isnan(raw.longitude);

    sub = raw(valid, :);

    % Convert geodetic GPS (Lat, Lon) to local Cartesian (X, Y) in meters relative to BTS
    m_per_deg_lat = 111320;
    m_per_deg_lon = 111320 * cosd(bs_lat);

    dx = (sub.longitude - bs_lon) * m_per_deg_lon;
    dy = (sub.latitude  - bs_lat) * m_per_deg_lat;
    dist = sqrt(dx.^2 + dy.^2);

    % Restrict to urban coverage radius (50m to 1500m)
    in_radius = (dist >= 50) & (dist <= 1500);
    sub = sub(in_radius, :);
    dx = dx(in_radius);
    dy = dy(in_radius);
    dist = dist(in_radius);

    % Spatial thinning: 15m grid binning to remove dwell points
    bin_x = round(dx / 15);
    bin_y = round(dy / 15);
    [~, u_idx] = unique([bin_x, bin_y], 'rows');

    sub = sub(u_idx, :);
    dx = dx(u_idx);
    dy = dy(u_idx);
    dist = dist(u_idx);

    fprintf('  Spatial grid-thinned road telemetry points: %d\n', height(sub));

    % Compute physics features
    logDist = log10(dist);
    azimuth = atan2d(dy, dx);

    % 4 Spatial Zones across Chania based on coordinate medians
    % Zone 1 (SW), Zone 2 (NW), Zone 3 (SE), Zone 4 (NE)
    med_x = median(dx);
    med_y = median(dy);

    zone_id = zeros(height(sub), 1);
    zone_id(dx <= med_x & dy <= med_y) = 1; % Zone 1: Southwest Corridor
    zone_id(dx <= med_x & dy >  med_y) = 2; % Zone 2: Northwest Corridor
    zone_id(dx >  med_x & dy <= med_y) = 3; % Zone 3: Southeast Corridor
    zone_id(dx >  med_x & dy >  med_y) = 4; % Zone 4: Northeast Corridor (Test Zone)

    outTable = table(sub.timestamp, sub.latitude, sub.longitude, ...
        dx, dy, dist, logDist, azimuth, zone_id, sub.cellID, sub.freq_dlink, sub.rssi, ...
        'VariableNames', {'Timestamp', 'Latitude', 'Longitude', 'X', 'Y', ...
                          'Distance', 'LogDist', 'Azimuth', 'RouteID', ...
                          'CellID', 'Frequency_MHz', 'RSRP_dBm'});

    % Ensure output directory exists
    [parentDir, ~, ~] = fileparts(outCsvPath);
    if ~isempty(parentDir) && ~exist(parentDir, 'dir')
        mkdir(parentDir);
    end

    writetable(outTable, outCsvPath);
    fprintf('>>> Successfully exported %d genuine field measurements to:\n    %s\n', height(outTable), outCsvPath);
    for z = 1:4
        fprintf('  Zone %d: %d samples (%.1f%%)\n', z, sum(zone_id == z), 100 * sum(zone_id == z) / height(outTable));
    end
end

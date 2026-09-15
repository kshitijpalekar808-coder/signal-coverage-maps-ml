%% MASTER BENCHMARK: GENUINE FIELD MEASUREMENTS (mySignals GSM Dataset)
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Source: mySignals GSM Dataset (Alimpertis et al., IEEE WCL 2014)
% Cellular Deployment: Base Station 6056x, Chania, Crete, Greece
% Carrier Frequency: 1860.2 MHz (GSM-1800 Downlink)
%
% Protocol: Strict Spatial Zone Holdout (Zero Data Leakage)
%   - Training: Zones 1, 2, 3 (Southwest, Northwest, Southeast - 662 samples)
%   - Testing:  Zone 4 (Northeast unvisited corridor - 236 samples)
% =========================================================================

function [resultsTable, models] = run_real_field_benchmark()
    projectRoot = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(projectRoot, 'src')));

    fprintf('======================================================================\n');
    fprintf('  REAL FIELD DATA BENCHMARK: mySignals GSM-1800 (Chania, Greece)\n');
    fprintf('======================================================================\n\n');

    csvPath = fullfile(projectRoot, 'data', 'real_mysignals_dataset.csv');
    if ~exist(csvPath, 'file')
        fprintf('>>> real_mysignals_dataset.csv not found. Generating from mySignals raw logs...\n');
        prepare_real_mysignals_dataset([], csvPath);
    end

    [trainData, testData, bsInfo, allData] = load_real_mysignals_data(csvPath, 4);

    fprintf('\n>>> Fitting Models on Real Field Training Telemetry (N = %d)...\n', height(trainData));

    % 1. 3GPP Theoretical Reference (No training)
    m1 = theoretical_propagation_model(bsInfo);
    p1 = m1.predict(testData);
    met1 = evaluate_predictions(testData.RSRP, p1);

    % 2. Empirical Log-Distance Path Loss (OLS)
    m2 = log_distance_model(trainData);
    p2 = m2.predict(testData);
    met2 = evaluate_predictions(testData.RSRP, p2);

    % 3. True kNN-IDW (p=2, k=30)
    m3 = knn_idw_model(trainData, 2.0, 30);
    p3 = m3.predict(testData);
    met3 = evaluate_predictions(testData.RSRP, p3);

    % 4. Natural Neighbor Interpolation
    m4 = natural_neighbor_model(trainData);
    p4 = m4.predict(testData);
    met4 = evaluate_predictions(testData.RSRP, p4);

    % 5. Random Forest (150 Trees)
    rng(42);
    m5 = random_forest_model(trainData, {'X', 'Y', 'Distance', 'LogDist', 'Azimuth'}, 150);
    p5 = m5.predict(testData);
    met5 = evaluate_predictions(testData.RSRP, p5);

    % 6. Proposed: Spatial ARD GPR [X, Y]
    rng(42);
    m6 = physics_informed_gpr_model(trainData, {'X', 'Y'});
    [p6, sigma_gpr] = m6.predict(testData);
    met6 = evaluate_predictions(testData.RSRP, p6);

    % Package benchmark table
    modelNames = {
        '1. 3GPP TR 38.901 Reference (No Training)';
        '2. Empirical Log-Distance Fit (OLS)';
        '3. True kNN-IDW (p=2, k=30)';
        '4. Natural Neighbor Interpolation';
        '5. Random Forest (150 Trees)';
        '6. Proposed: Spatial ARD GPR [X, Y]'
    };

    allMets = [met1; met2; met3; met4; met5; met6];
    resultsTable = table(modelNames, ...
        [allMets.RMSE]', [allMets.MAE]', [allMets.R2]', [allMets.MaxAE]', [allMets.P95AE]', ...
        'VariableNames', {'Model', 'RMSE_dB', 'MAE_dB', 'R2_Score', 'MaxAE_dB', 'P95AE_dB'});

    fprintf('\n======================================================================\n');
    fprintf('  REAL FIELD BENCHMARK RESULTS (mySignals GSM Dataset, Held-Out Zone 4)\n');
    fprintf('======================================================================\n');
    disp(resultsTable);

    % Save results CSV
    outCsv = fullfile(projectRoot, 'results', 'real_field_benchmark_results.csv');
    writetable(resultsTable, outCsv);
    fprintf('Saved real field benchmark CSV to: %s\n', outCsv);

    % Save duplicate for backwards compatibility
    writetable(resultsTable, fullfile(projectRoot, 'results', 'real_data_benchmark_results.csv'));

    % Generate Publication Figure
    fig = figure('Position', [100, 100, 1200, 900], 'Color', 'w', 'Visible', 'off');

    % Panel 1: GPS Field Tracks and Spatial Holdout Partition
    subplot(2, 2, 1);
    scatter(trainData.X, trainData.Y, 20, trainData.RSRP, 'filled');
    hold on;
    scatter(testData.X, testData.Y, 30, testData.RSRP, 'd', 'filled', 'MarkerEdgeColor', 'k');
    plot(0, 0, 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y');
    colormap(gca, 'parula');
    cb1 = colorbar; cb1.Label.String = 'RSRP (dBm)';
    title('(a) Real Field Tracks: Train (circle) vs Test Zone 4 (diamond)', 'FontWeight', 'bold');
    xlabel('X relative to BTS (m)'); ylabel('Y relative to BTS (m)');
    grid on; axis equal; xlim([-600, 1400]); ylim([-500, 1300]);

    % Panel 2: Continuous 2D Reconstructed Coverage Surface
    subplot(2, 2, 2);
    [X_grid, Y_grid] = meshgrid(-600:25:1400, -500:25:1300);
    gridTable = table(X_grid(:), Y_grid(:), 'VariableNames', {'X', 'Y'});
    [grid_pred, grid_sigma] = m6.predict(gridTable);
    surf_pred = reshape(grid_pred, size(X_grid));
    surf_sigma = reshape(grid_sigma, size(X_grid));

    pcolor(X_grid, Y_grid, surf_pred); shading interp;
    hold on;
    plot(0, 0, 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y');
    colormap(gca, 'parula');
    cb2 = colorbar; cb2.Label.String = 'Predicted RSRP (dBm)';
    title('(b) Spatial ARD GPR Reconstructed Coverage Map', 'FontWeight', 'bold');
    xlabel('X (m)'); ylabel('Y (m)');
    grid on; axis equal;

    % Panel 3: Spatial Latent Uncertainty (sigma_latent)
    subplot(2, 2, 3);
    pcolor(X_grid, Y_grid, surf_sigma); shading interp;
    hold on;
    scatter(allData.X, allData.Y, 6, [0.3 0.3 0.3], 'filled');
    plot(0, 0, 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y');
    colormap(gca, 'hot');
    cb3 = colorbar; cb3.Label.String = 'Uncertainty \sigma_{latent} (dB)';
    title('(c) GPR Latent Uncertainty \sigma_{latent}(x,y)', 'FontWeight', 'bold');
    xlabel('X (m)'); ylabel('Y (m)');
    grid on; axis equal;

    % Panel 4: Error Residuals on Unvisited Test Corridor
    subplot(2, 2, 4);
    errors = testData.RSRP - p6;
    stem(1:length(errors), errors, 'filled', 'MarkerSize', 4, 'Color', [0.15 0.35 0.7]);
    hold on;
    yline(0, 'k--', 'LineWidth', 1.2);
    yline(met6.RMSE, 'r--', sprintf('+RMSE (%.2f dB)', met6.RMSE));
    yline(-met6.RMSE, 'r--', sprintf('-RMSE (%.2f dB)', met6.RMSE));
    title('(d) Held-Out Test Route Residuals (Zone 4)', 'FontWeight', 'bold');
    xlabel('Test Telemetry Sample Index'); ylabel('Prediction Residual (dB)');
    grid on; ylim([-20, 20]);

    figPath1 = fullfile(projectRoot, 'results', 'figures', 'real_field_coverage_reconstruction.png');
    figPath2 = fullfile(projectRoot, 'results', 'real_data_coverage_reconstruction.png');
    exportgraphics(fig, figPath1, 'Resolution', 300);
    exportgraphics(fig, figPath2, 'Resolution', 300);
    close(fig);
    fprintf('Exported publication figure to: %s\n', figPath1);

    models.theoModel = m1;
    models.logModel  = m2;
    models.idwModel  = m3;
    models.natModel  = m4;
    models.rfModel   = m5;
    models.gprModel  = m6;
end

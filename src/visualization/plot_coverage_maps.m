%% VISUALIZATION: PUBLICATION-QUALITY 2D COVERAGE & UNCERTAINTY MAPS
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Function: plot_coverage_maps
% Generates a 4-panel comprehensive visual summary:
%   Panel (a): Urban Street Route Layout (Train Routes 1-9 vs Unseen Routes 10-12)
%   Panel (b): Continuous Physics-Informed GPR Reconstructed Coverage Map (dBm)
%   Panel (c): GPR Latent Spatial Field Uncertainty sigma_latent(x, y)
%   Panel (d): Held-Out Test Error Residual Map & Correlation Scatter Plot
% =========================================================================

function fig = plot_coverage_maps(fullGrid, trainData, testData, gprModel, bsInfo, saveOutputs)
    if nargin < 6 || isempty(saveOutputs), saveOutputs = true; end

    gridSize = round(sqrt(height(fullGrid)));
    X_mat = reshape(fullGrid.X, gridSize, gridSize);
    Y_mat = reshape(fullGrid.Y, gridSize, gridSize);

    % Predict GPR mean and latent field uncertainty across entire 2D urban window
    [pred_mean, sigma_latent, ~] = gprModel.predict(fullGrid);
    map_rsrp = reshape(pred_mean, gridSize, gridSize);
    map_sigma = reshape(sigma_latent, gridSize, gridSize);

    % Predict on held-out test routes
    [test_pred, test_sigma] = gprModel.predict(testData);
    test_err = abs(testData.RSRP - test_pred);

    c_limits = [-115, -55]; % Standard cellular RSRP visualization limits in dBm

    fig = figure('Name', 'Physics-Informed Coverage & Uncertainty Reconstruction', ...
                 'Color', 'w', 'Position', [80, 80, 1280, 920]);

    % ---------------------------------------------------------------------
    % Panel (a): Spatial Drive-Test Layout (Route Holdout Partition)
    % ---------------------------------------------------------------------
    subplot(2, 2, 1);
    hold on;
    scatter(trainData.X, trainData.Y, 18, [0.2 0.45 0.85], 'filled', ...
        'DisplayName', sprintf('Training Routes 1-9 (%d pts)', height(trainData)));
    scatter(testData.X, testData.Y, 24, [0.85 0.2 0.2], 'filled', ...
        'DisplayName', sprintf('Held-Out Test Routes 10-12 (%d pts)', height(testData)));
    plot(bsInfo.localX, bsInfo.localY, 'p', 'MarkerSize', 16, ...
        'MarkerFaceColor', [1.0 0.8 0.0], 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Serving Base Station (%d MHz)', round(bsInfo.freq_GHz*1000)));

    title('(a) Urban Drive-Test Routes & Spatial Holdout', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    xlabel('East Coordinate X (m)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    ylabel('North Coordinate Y (m)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    xlim([0, 1000]); ylim([0, 1000]); axis square; grid on; box on;
    legend('Location', 'northeast', 'TextColor', 'k', 'FontSize', 9);
    set(gca, 'XColor', 'k', 'YColor', 'k');

    % ---------------------------------------------------------------------
    % Panel (b): Physics-Informed GPR Reconstructed Continuous Coverage
    % ---------------------------------------------------------------------
    subplot(2, 2, 2);
    imagesc([0, 1000], [0, 1000], map_rsrp);
    set(gca, 'YDir', 'normal');
    colormap(gca, turbo);
    clim(c_limits);
    cb1 = colorbar;
    ylabel(cb1, 'Predicted RSRP (dBm)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    set(cb1, 'Color', 'k');
    hold on;
    plot(bsInfo.localX, bsInfo.localY, 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y', 'LineWidth', 1.5);
    
    title('(b) Reconstructed Continuous 2D Coverage Field \mu(x,y)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    xlabel('East Coordinate X (m)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    ylabel('North Coordinate Y (m)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    xlim([0, 1000]); ylim([0, 1000]); axis square; box on;
    set(gca, 'XColor', 'k', 'YColor', 'k');

    % ---------------------------------------------------------------------
    % Panel (c): Latent Spatial Field Uncertainty sigma_latent(x,y)
    % ---------------------------------------------------------------------
    subplot(2, 2, 3);
    imagesc([0, 1000], [0, 1000], map_sigma);
    set(gca, 'YDir', 'normal');
    colormap(gca, hot);
    cb2 = colorbar;
    ylabel(cb2, 'Latent Field Uncertainty \sigma_{latent} (dB)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    set(cb2, 'Color', 'k');
    hold on;
    plot(trainData.X, trainData.Y, 'c.', 'MarkerSize', 3, 'DisplayName', 'Sampled Roads');
    plot(bsInfo.localX, bsInfo.localY, 'yp', 'MarkerSize', 14, 'MarkerFaceColor', 'y', 'LineWidth', 1.5);

    title('(c) Latent Spatial Field Uncertainty \sigma_{latent}(x,y)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    xlabel('East Coordinate X (m)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    ylabel('North Coordinate Y (m)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    xlim([0, 1000]); ylim([0, 1000]); axis square; box on;
    set(gca, 'XColor', 'k', 'YColor', 'k');

    % ---------------------------------------------------------------------
    % Panel (d): Held-Out Test Correlation & Error Residual Plot
    % ---------------------------------------------------------------------
    subplot(2, 2, 4);
    scatter(testData.RSRP, test_pred, 32, test_err, 'filled');
    colormap(gca, parula);
    cb3 = colorbar;
    ylabel(cb3, 'Absolute Error |y - \mu| (dB)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    set(cb3, 'Color', 'k');
    hold on;
    minVal = min([testData.RSRP; test_pred]) - 2;
    maxVal = max([testData.RSRP; test_pred]) + 2;
    plot([minVal, maxVal], [minVal, maxVal], 'r--', 'LineWidth', 2, 'DisplayName', 'Ideal 1:1 Fit');

    rmse_val = sqrt(mean((testData.RSRP - test_pred).^2));
    r2_val = 1 - sum((testData.RSRP - test_pred).^2) / sum((testData.RSRP - mean(testData.RSRP)).^2);
    
    title(sprintf('(d) Held-Out Corridors (RMSE: %.2f dB, R^2: %.3f)', rmse_val, r2_val), ...
        'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    xlabel('Measured Ground Truth RSRP (dBm)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    ylabel('Physics-Informed GPR Prediction (dBm)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    xlim([minVal, maxVal]); ylim([minVal, maxVal]); axis square; grid on; box on;
    legend('Location', 'northwest', 'TextColor', 'k', 'FontSize', 9);
    set(gca, 'XColor', 'k', 'YColor', 'k');

    if saveOutputs
        if ~exist('results', 'dir'), mkdir('results'); end
        if ~exist(fullfile('results', 'figures'), 'dir'), mkdir(fullfile('results', 'figures')); end

        saveas(fig, fullfile('results', 'figures', 'coverage_reconstruction_map.png'));
        saveas(fig, fullfile('results', 'real_data_coverage_reconstruction.png')); % Backward compatibility
        fprintf('Saved coverage reconstruction figure to results/figures/coverage_reconstruction_map.png\n');
    end
end

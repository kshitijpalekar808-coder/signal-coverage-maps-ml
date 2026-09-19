%% VISUALIZATION: CONTROLLED SPARSITY STRESS TEST PLOT
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Function: plot_sparsity_analysis
% Plots reconstruction error (RMSE in dB) across sampling densities (2% to 40%)
% comparing kNN-IDW, Natural Neighbor, Empirical Log-Distance, Random Forest,
% and Physics-Informed GPR.
% =========================================================================

function fig = plot_sparsity_analysis(sparsityTable, saveOutputs)
    if nargin < 2 || isempty(saveOutputs), saveOutputs = true; end

    fig = figure('Name', 'Controlled Sparsity Stress Test', 'Color', 'w', 'Position', [120, 120, 920, 520]);

    densities = sparsityTable.SamplingDensity_Pct;

    plot(densities, sparsityTable.LogDist_RMSE_dB, '--', 'Color', [0.6 0.2 0.8], 'LineWidth', 2.0, ...
        'Marker', 'd', 'MarkerSize', 7, 'DisplayName', 'Empirical Log-Distance (OLS)');
    hold on;
    plot(densities, sparsityTable.RandomForest_RMSE_dB, '-.', 'Color', [0.2 0.7 0.3], 'LineWidth', 2.0, ...
        'Marker', '^', 'MarkerSize', 7, 'DisplayName', 'Random Forest (150 Trees)');
    plot(densities, sparsityTable.IDW_RMSE_dB, '-.', 'Color', [0.85 0.45 0.1], 'LineWidth', 2.0, ...
        'Marker', 'o', 'MarkerSize', 7, 'DisplayName', 'True kNN-IDW (p=2, k=30)');
    plot(densities, sparsityTable.GPR_RMSE_dB, '-s', 'Color', [0.1 0.45 0.85], 'LineWidth', 2.6, ...
        'MarkerSize', 8, 'MarkerFaceColor', [0.1 0.45 0.85], 'DisplayName', 'Proposed: Spatial ARD GPR [X, Y]');

    xlabel('Measurement Sampling Density (% of Urban Grid)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    ylabel('Reconstruction RMSE (dB)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    title('Controlled Synthetic Sparsity Experiment: Error vs. Measurement Density', ...
        'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');
    legend('Location', 'northeast', 'TextColor', 'k', 'FontSize', 10);
    grid on; box on; set(gca, 'XColor', 'k', 'YColor', 'k');
    xlim([0, 42]);

    if saveOutputs
        if ~exist('results', 'dir'), mkdir('results'); end
        if ~exist(fullfile('results', 'figures'), 'dir'), mkdir(fullfile('results', 'figures')); end

        saveas(fig, fullfile('results', 'figures', 'sparsity_stress_test.png'));
        saveas(fig, fullfile('results', 'sparsity_stress_test.png')); % Root copy
        fprintf('Saved sparsity stress test figure to results/figures/sparsity_stress_test.png\n');
    end
end

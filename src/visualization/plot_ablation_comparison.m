%% VISUALIZATION: FEATURE ABLATION COMPARISON BAR CHART
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Function: plot_ablation_comparison
% Generates a grouped publication-quality bar chart comparing:
%   - Config A: Spatial Only [X, Y]
%   - Config B: Radio Physics Only [Distance, LogDist, Azimuth]
%   - Config C: Full Hybrid [X, Y, Distance, LogDist, Azimuth]
% Across both Physics-Informed GPR and Random Forest (150 Trees).
% =========================================================================

function fig = plot_ablation_comparison(ablationTable, saveOutputs)
    if nargin < 2 || isempty(saveOutputs), saveOutputs = true; end

    fig = figure('Name', 'Feature Ablation Study Comparison', 'Color', 'w', 'Position', [120, 120, 960, 520]);

    barData = [ablationTable.GPR_RMSE_dB, ablationTable.RF_RMSE_dB];
    b = bar(barData, 'grouped');
    b(1).FaceColor = [0.15 0.50 0.85];
    b(2).FaceColor = [0.25 0.70 0.45];

    set(gca, 'XTickLabel', {'Config A: Spatial Only [X, Y]', ...
                            'Config B: Physics Only [Dist, LogD, Az]', ...
                            'Config C: Full Hybrid [All 5 Features]'}, ...
             'FontSize', 11, 'XColor', 'k', 'YColor', 'k');

    ylabel('Held-Out Test RMSE (dB)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    title('Feature Ablation: Impact of Radio-Physics Feature Engineering on Unseen Routes', ...
        'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');
    legend({'Physics-Informed GPR', 'Random Forest (150 Trees)'}, 'Location', 'northeast', 'TextColor', 'k', 'FontSize', 11);
    grid on; box on;

    % Annotate numerical values on top of bars
    for i = 1:size(barData, 1)
        text(i - 0.15, barData(i, 1) + 0.15, sprintf('%.2f dB', barData(i, 1)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.1 0.3 0.6]);
        text(i + 0.15, barData(i, 2) + 0.15, sprintf('%.2f dB', barData(i, 2)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.1 0.5 0.2]);
    end
    ylim([0, max(barData(:)) * 1.15]);

    if saveOutputs
        if ~exist('results', 'dir'), mkdir('results'); end
        if ~exist(fullfile('results', 'figures'), 'dir'), mkdir(fullfile('results', 'figures')); end

        saveas(fig, fullfile('results', 'figures', 'ablation_study_comparison.png'));
        saveas(fig, fullfile('results', 'ablation_study_comparison.png')); % Root copy
        fprintf('Saved feature ablation figure to results/figures/ablation_study_comparison.png\n');
    end
end

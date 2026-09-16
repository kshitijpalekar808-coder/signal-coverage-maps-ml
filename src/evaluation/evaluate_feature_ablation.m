%% EVALUATION: FEATURE ABLATION STUDY (PHYSICS VS SPATIAL COORDINATES)
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Function: evaluate_feature_ablation
% Quantifies the predictive value of domain-specific radio propagation
% features versus spatial coordinates by evaluating:
%   - Config A: Spatial Coordinates Only [X, Y]
%   - Config B: Radio Physics Features Only [Distance, LogDist, Azimuth]
%   - Config C: Full Hybrid Physics-Informed Model [All 5 Features]
% Across both Physics-Informed GPR and Random Forest (150 Trees) on
% unvisited held-out test corridors (Routes 10-12).
% =========================================================================

function [ablationTable, fig] = evaluate_feature_ablation(trainData, testData, saveOutputs)
    if nargin < 3 || isempty(saveOutputs)
        saveOutputs = true;
    end

    if nargin < 1 || isempty(trainData) || nargin < 2 || isempty(testData)
        [trainData, testData] = load_urban_drive_test_data([], 'route_holdout', [10, 11, 12]);
    end

    ablationConfigs = {
        {'X', 'Y'}, ...
        {'Distance', 'LogDist', 'Azimuth'}, ...
        {'X', 'Y', 'Distance', 'LogDist', 'Azimuth'}
    };
    configNames = {
        'Config A: Spatial Only [X, Y]'; ...
        'Config B: Radio Physics Only [Distance, LogDist, Azimuth]'; ...
        'Config C: Full Hybrid [All 5 Features]'
    };

    numConfigs = length(ablationConfigs);
    gpr_abl_rmse = zeros(numConfigs, 1);
    gpr_abl_r2   = zeros(numConfigs, 1);
    rf_abl_rmse  = zeros(numConfigs, 1);
    rf_abl_r2    = zeros(numConfigs, 1);

    fprintf('======================================================================\n');
    fprintf('  RUNNING FEATURE ABLATION STUDY (Physics vs Spatial Coordinates)\n');
    fprintf('======================================================================\n');

    for i = 1:numConfigs
        feats = ablationConfigs{i};
        fprintf('Evaluating %s...\n', configNames{i});

        % ARD GPR with independent hyperparameter optimization on train set
        gpr_abl = physics_informed_gpr_model(trainData, feats);
        [p_gpr, ~, ~] = gpr_abl.predict(testData);
        m_gpr = evaluate_predictions(testData.RSRP, p_gpr);
        gpr_abl_rmse(i) = m_gpr.RMSE;
        gpr_abl_r2(i)   = m_gpr.R2;

        % Random Forest (150 Trees)
        rf_abl = random_forest_model(trainData, feats, 150, 42);
        p_rf = rf_abl.predict(testData);
        m_rf = evaluate_predictions(testData.RSRP, p_rf);
        rf_abl_rmse(i)  = m_rf.RMSE;
        rf_abl_r2(i)    = m_rf.R2;

        fprintf('  -> GPR: RMSE = %.2f dB, R2 = %.3f | RF: RMSE = %.2f dB, R2 = %.3f\n', ...
            gpr_abl_rmse(i), gpr_abl_r2(i), rf_abl_rmse(i), rf_abl_r2(i));
    end

    ablationTable = table(configNames, gpr_abl_rmse, gpr_abl_r2, rf_abl_rmse, rf_abl_r2, ...
        'VariableNames', {'FeatureConfiguration', 'GPR_RMSE_dB', 'GPR_R2', 'RF_RMSE_dB', 'RF_R2'});

    fprintf('\n=== FEATURE ABLATION BENCHMARK RESULTS ===\n');
    disp(ablationTable);

    fig = [];
    if saveOutputs
        if ~exist('results', 'dir'), mkdir('results'); end
        if ~exist(fullfile('results', 'figures'), 'dir'), mkdir(fullfile('results', 'figures')); end

        writetable(ablationTable, fullfile('results', 'ablation_study_results.csv'));
        fig = plot_ablation_comparison(ablationTable, true);
    end
end

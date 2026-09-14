%% EVALUATION: CONTROLLED SYNTHETIC SPARSITY STRESS TEST
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Description:
% Evaluates reconstruction accuracy (RMSE in dB) across sampling densities
% (e.g. 2%, 5%, 8%, 10%, 20%, 40%) under known continuous 2D ground truth
% (N = 10,201 points). Demonstrates that physics-informed ML and spatial
% methods maintain high accuracy even with minimal drive-test probes.
% =========================================================================

function [sparsityTable, fig] = run_sparsity_analysis(sparsities, saveOutputs)
    if nargin < 1 || isempty(sparsities)
        sparsities = [0.02, 0.05, 0.08, 0.10, 0.20, 0.40];
    end
    if nargin < 2 || isempty(saveOutputs)
        saveOutputs = true;
    end

    numLevels = length(sparsities);
    sp_idw_rmse = zeros(numLevels, 1);
    sp_log_rmse = zeros(numLevels, 1);
    sp_rf_rmse  = zeros(numLevels, 1);
    sp_gpr_rmse = zeros(numLevels, 1);

    fprintf('======================================================================\n');
    fprintf('  RUNNING CONTROLLED SPARSITY STRESS TEST (%d DENSITY LEVELS)\n', numLevels);
    fprintf('======================================================================\n');

    for i = 1:numLevels
        s = sparsities(i);
        fprintf('Evaluating Density Level %d/%d (%.1f%% Sampling)...\n', i, numLevels, s * 100);

        % Generate synthetic continuous ground truth with sparse sampling
        [tr_s, te_s, ~, ~, ~, ~] = generate_sparsity_dataset(s);

        % 1. True kNN-IDW (p=2, k=30)
        idw_s = knn_idw_model(tr_s, 2, 30);
        p_idw_s = idw_s.predict(te_s);

        % 2. Empirical Log-Distance Path Loss
        log_s = log_distance_model(tr_s);
        p_log_s = log_s.predict(te_s);

        % 3. Random Forest (150 Trees)
        rf_s = random_forest_model(tr_s, {'X', 'Y', 'Distance', 'LogDist', 'Azimuth'}, 150, 42);
        p_rf_s = rf_s.predict(te_s);

        % 4. Proposed: Spatial ARD GPR [X, Y] (Matérn 5/2, MLL optimized on train)
        gpr_s = physics_informed_gpr_model(tr_s, {'X', 'Y'});
        [p_gpr_s, ~, ~] = gpr_s.predict(te_s);

        sp_idw_rmse(i) = sqrt(mean((te_s.RSRP - p_idw_s).^2));
        sp_log_rmse(i) = sqrt(mean((te_s.RSRP - p_log_s).^2));
        sp_rf_rmse(i)  = sqrt(mean((te_s.RSRP - p_rf_s).^2));
        sp_gpr_rmse(i) = sqrt(mean((te_s.RSRP - p_gpr_s).^2));

        fprintf('  -> Density %.1f%%: IDW=%.2f dB, LogDist=%.2f dB, RF=%.2f dB, GPR=%.2f dB\n', ...
            s * 100, sp_idw_rmse(i), sp_log_rmse(i), sp_rf_rmse(i), sp_gpr_rmse(i));
    end

    sparsityTable = table((sparsities * 100)', sp_idw_rmse, sp_log_rmse, sp_gpr_rmse, sp_rf_rmse, ...
        'VariableNames', {'SamplingDensity_Pct', 'IDW_RMSE_dB', 'LogDist_RMSE_dB', 'GPR_RMSE_dB', 'RandomForest_RMSE_dB'});

    fprintf('\n=== SPARSITY STRESS TEST RESULTS ===\n');
    disp(sparsityTable);

    fig = [];
    if saveOutputs
        thisDir = fileparts(which('run_sparsity_analysis'));
        projectRoot = fileparts(fileparts(thisDir));
        if isempty(projectRoot), projectRoot = pwd; end

        resDir = fullfile(projectRoot, 'results');
        figDir = fullfile(projectRoot, 'results', 'figures');
        if ~exist(resDir, 'dir'), mkdir(resDir); end
        if ~exist(figDir, 'dir'), mkdir(figDir); end

        writetable(sparsityTable, fullfile(resDir, 'sparsity_stress_results.csv'));
        writetable(sparsityTable, fullfile(resDir, 'sparsity_results.csv'));
        fig = plot_sparsity_analysis(sparsityTable, true);
    end
end

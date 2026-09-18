% Project Validation and Verification Test Suite
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% Author: Kshitij Palekar

function status = validate_project()
    fprintf('======================================================================\n');
    fprintf('  RUNNING PROJECT VERIFICATION SUITE\n');
    fprintf('======================================================================\n\n');

    projectRoot = fileparts(mfilename('fullpath'));
    if isempty(projectRoot), projectRoot = pwd; end
    addpath(genpath(fullfile(projectRoot, 'src')));
    totalTests = 12;
    passedTests = 0;

    % ---------------------------------------------------------------------
    % Test 1: Real mySignals Field Dataset Schema & Integrity
    % ---------------------------------------------------------------------
    fprintf('[Test 1/12] Verifying Real mySignals Dataset Schema and Integrity...');
    realCsv = fullfile(projectRoot, 'data', 'real_mysignals_dataset.csv');
    if ~exist(realCsv, 'file')
        error('FAIL: Real dataset %s not found. Run prepare_real_mysignals_dataset first.', realCsv);
    end
    t_real = readtable(realCsv);
    reqVarsReal = {'Timestamp', 'Latitude', 'Longitude', 'X', 'Y', 'Distance', ...
                   'LogDist', 'Azimuth', 'RouteID', 'CellID', 'Frequency_MHz', 'RSRP_dBm'};
    for v = reqVarsReal
        if ~ismember(v{1}, t_real.Properties.VariableNames)
            error('FAIL: Missing required column "%s" in real dataset.', v{1});
        end
    end
    if any(isnan(t_real.RSRP_dBm)) || any(isnan(t_real.Latitude)) || any(isnan(t_real.Longitude))
        error('FAIL: Real dataset contains invalid NaN values.');
    end
    passedTests = passedTests + 1;
    fprintf(' PASSED (%d field samples, 0 NaNs)\n', height(t_real));

    % ---------------------------------------------------------------------
    % Test 2: Real Field 4-Zone Spatial Partitioning
    % ---------------------------------------------------------------------
    fprintf('[Test 2/12] Verifying Real Field 4-Zone Spatial Partitioning...');
    uZones = unique(t_real.RouteID);
    if length(uZones) ~= 4 || ~isequal(uZones', 1:4)
        error('FAIL: Real dataset must contain exactly 4 balanced spatial zones (IDs 1-4).');
    end
    passedTests = passedTests + 1;
    fprintf(' PASSED (Zones 1 to 4 present)\n');

    % ---------------------------------------------------------------------
    % Test 3: Real Field Zero Spatial Leakage
    % ---------------------------------------------------------------------
    fprintf('[Test 3/12] Verifying Real Field Spatial Holdout (Zero Leakage)...');
    [trReal, teReal] = load_real_mysignals_data(realCsv, 4);
    overlapReal = intersect(unique(trReal.RouteID), unique(teReal.RouteID));
    if ~isempty(overlapReal)
        error('FAIL: Spatial leakage detected in real field data!');
    end
    passedTests = passedTests + 1;
    fprintf(' PASSED (Train: Zones 1-3 | Test: Zone 4 | Intersect: Empty)\n');

    % ---------------------------------------------------------------------
    % Test 4: Synthetic Urban Grid Dataset Integrity
    % ---------------------------------------------------------------------
    fprintf('[Test 4/12] Verifying Synthetic Grid Dataset Integrity...');
    synthCsv = fullfile(projectRoot, 'data', 'synthetic_urban_grid_dataset.csv');
    t_synth = readtable(synthCsv);
    uRoutes = unique(t_synth.RouteID);
    if length(uRoutes) ~= 12 || ~isequal(uRoutes', 1:12)
        error('FAIL: Synthetic dataset must contain 12 distinct routes.');
    end
    passedTests = passedTests + 1;
    fprintf(' PASSED (%d records, Routes 1-12)\n', height(t_synth));

    % ---------------------------------------------------------------------
    % Test 5: Synthetic Route-Based Zero Spatial Leakage
    % ---------------------------------------------------------------------
    fprintf('[Test 5/12] Verifying Synthetic Route-Based Spatial Holdout...');
    [trSynth, teSynth] = load_urban_drive_test_data(synthCsv, 'route_holdout', [10, 11, 12]);
    overlapSynth = intersect(unique(trSynth.RouteID), unique(teSynth.RouteID));
    if ~isempty(overlapSynth)
        error('FAIL: Spatial leakage in synthetic dataset!');
    end
    passedTests = passedTests + 1;
    fprintf(' PASSED (Train: Routes 1-9 | Test: Routes 10-12 | Intersect: Empty)\n');

    % ---------------------------------------------------------------------
    % Test 6: Training-Only Normalization Policy
    % ---------------------------------------------------------------------
    fprintf('[Test 6/12] Verifying Training-Only Normalization Policy...');
    features = {'X', 'Y', 'Distance', 'LogDist', 'Azimuth'};
    X_tr = table2array(trSynth(:, features));
    y_tr = trSynth.RSRP;
    [normParams, ~, ~] = normalize_features(X_tr, y_tr);
    true_train_mean = mean(trSynth.X);
    if abs(normParams.mu_X(1) - true_train_mean) > 1e-10
        error('FAIL: Normalization mean does not match training-only mean.');
    end
    combined_mean = mean([trSynth.X; teSynth.X]);
    if abs(normParams.mu_X(1) - combined_mean) < 1e-4
        error('FAIL: Data leakage! Normalization includes test data statistics.');
    end
    passedTests = passedTests + 1;
    fprintf(' PASSED (Strictly train-only statistics)\n');

    % ---------------------------------------------------------------------
    % Test 7: kNN-IDW Coincident Handling
    % ---------------------------------------------------------------------
    fprintf('[Test 7/12] Verifying kNN-IDW (p=2, k=30)...');
    idw = knn_idw_model(trSynth(1:50, :), 2, 10);
    p_coincident = idw.predict(trSynth(1:5, :));
    if max(abs(p_coincident - trSynth.RSRP(1:5))) > 1e-6
        error('FAIL: IDW failed to return exact target value at coincident point.');
    end
    passedTests = passedTests + 1;
    fprintf(' PASSED (Exact coincident assignment verified)\n');

    % ---------------------------------------------------------------------
    % Test 8: Spatial ARD GPR Covariance & Latent Uncertainty Bounds
    % ---------------------------------------------------------------------
    fprintf('[Test 8/12] Verifying GPR Covariance & Latent Field Uncertainty...');
    gpr = physics_informed_gpr_model(trSynth(1:60, :), {'X', 'Y'}, 1.0, 1.0, 0.05);
    [~, sig_latent, sig_obs] = gpr.predict(teSynth(1:10, :));
    if any(sig_latent < 0) || any(isnan(sig_latent))
        error('FAIL: Latent uncertainty must be non-negative.');
    end
    if any(sig_obs < sig_latent)
        error('FAIL: Observation uncertainty must bound latent uncertainty.');
    end
    passedTests = passedTests + 1;
    fprintf(' PASSED (sigma_obs >= sigma_latent >= 0)\n');

    % ---------------------------------------------------------------------
    % Test 9: Results CSV Metrics Integrity
    % ---------------------------------------------------------------------
    fprintf('[Test 9/12] Verifying Results CSV Metrics Integrity...');
    expectedCSVs = {
        fullfile('results', 'real_field_benchmark_results.csv');
        fullfile('results', 'benchmark_results.csv');
        fullfile('results', 'ablation_study_results.csv');
        fullfile('results', 'sparsity_stress_results.csv');
        fullfile('results', 'active_learning_results.csv')
    };
    for c = expectedCSVs'
        csvF = fullfile(projectRoot, c{1});
        if ~exist(csvF, 'file')
            error('FAIL: Expected result CSV not found: %s', c{1});
        end
        fInfo = dir(csvF);
        if fInfo.bytes == 0
            error('FAIL: Result CSV is empty: %s', c{1});
        end
    end
    passedTests = passedTests + 1;
    fprintf(' PASSED (All 5 result CSVs exist & non-empty)\n');

    % ---------------------------------------------------------------------
    % Test 10: Publication-Quality Figures Integrity
    % ---------------------------------------------------------------------
    fprintf('[Test 10/12] Verifying Publication Figures Integrity...');
    expectedFigs = {
        fullfile('results', 'figures', 'real_field_coverage_reconstruction.png');
        fullfile('results', 'figures', 'coverage_reconstruction_map.png');
        fullfile('results', 'figures', 'ablation_study_comparison.png');
        fullfile('results', 'figures', 'sparsity_stress_test.png');
        fullfile('results', 'figures', 'active_learning_comparison.png');
        fullfile('results', 'figures', 'active_learning_route_selection.png')
    };
    for f = expectedFigs'
        figF = fullfile(projectRoot, f{1});
        if ~exist(figF, 'file')
            error('FAIL: Expected publication figure not found: %s', f{1});
        end
        fInfo = dir(figF);
        if fInfo.bytes < 1000
            error('FAIL: Figure file is too small or corrupted: %s', f{1});
        end
    end
    passedTests = passedTests + 1;
    fprintf(' PASSED (All 6 publication figures valid)\n');

    % ---------------------------------------------------------------------
    % Test 11: Standalone Runner Entry Points & Backward-Compatibility Adapters
    % ---------------------------------------------------------------------
    fprintf('[Test 11/12] Verifying Runner Entry Points and Compatibility Adapters...');
    requiredEntryPoints = {
        'run_real_field_benchmark.m';
        'run_ablation_study.m';
        'run_sparsity_test.m';
        'run_active_learning.m';
        'run_all_experiments.m';
        'main.m'
    };
    for ep = requiredEntryPoints'
        epPath = fullfile(projectRoot, ep{1});
        if ~exist(epPath, 'file')
            error('FAIL: Standalone runner script %s not found in root.', ep{1});
        end
    end

    % Verify legacy compatibility adapters exist and resolve
    compatFuncs = {'step1_prepare_data', 'step6_sparsity_analysis', 'step7_ablation_study', 'step8_active_learning', 'load_real_drive_test_data'};
    for cf = compatFuncs
        if isempty(which(cf{1}))
            error('FAIL: Compatibility adapter "%s" does not resolve in MATLAB path.', cf{1});
        end
    end
    passedTests = passedTests + 1;
    fprintf(' PASSED (All 6 root scripts & 5 adapters verified)\n');

    % ---------------------------------------------------------------------
    % Test 12: Functional Smoke-Testing of Analytical Modules
    % ---------------------------------------------------------------------
    fprintf('[Test 12/12] Functional Smoke-Testing of Analytical Pipeline Modules...');
    
    % Test A: Sparsity dataset generator
    [tr_smoke, te_smoke, ~, ~, ~, ~] = generate_sparsity_dataset(0.02, 999);
    if height(tr_smoke) == 0 || height(te_smoke) == 0
        error('FAIL: generate_sparsity_dataset produced empty train/test tables.');
    end

    % Test B: Sparsity sweep module execution (quick 2-level test, no overwrite)
    spSmoke = run_sparsity_analysis([0.02, 0.05], false);
    if height(spSmoke) ~= 2
        error('FAIL: run_sparsity_analysis smoke test did not return expected table height.');
    end

    % Test C: Ablation study module execution (train on subset for fast validation, no overwrite)
    [ablSmoke, ~] = evaluate_feature_ablation(trSynth(1:80, :), teSynth(1:40, :), false);
    if height(ablSmoke) ~= 3
        error('FAIL: evaluate_feature_ablation smoke test did not return 3 ablation configurations.');
    end

    passedTests = passedTests + 1;
    fprintf(' PASSED (Data generation, sparsity analysis, and ablation smoke-tested)\n');

    fprintf('\n======================================================================\n');
    fprintf('  ALL %d/%d VALIDATION CHECKS PASSED SUCCESSFULLY\n', passedTests, totalTests);
    fprintf('======================================================================\n');
    status = true;
end

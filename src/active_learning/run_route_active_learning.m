%% ACTIVE LEARNING: ROUTE-AWARE DRIVE-TEST MEASUREMENT OPTIMIZATION
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Description:
% Evaluates whether uncertainty-guided adaptive street selection reduces
% the number of drive-test routes required to achieve high coverage accuracy:
%
% Methodology & Experimental Protocol:
% -------------------------------------
% 1. Fixed Unseen Test Routes: Routes 10, 11, 12 (Held out, 0% leakage).
% 2. Candidate Route Pool: Routes 1 to 9.
% 3. Strategy 1 (Active GPR Uncertainty): Evaluates latent field uncertainty
%    sigma_latent(x, y) on unvisited candidate streets and directs the vehicle
%    to the highest-uncertainty route.
% 4. Strategy 2 (Passive Random Sampling): Evaluated over N = 20 repeated
%    independent random trials for statistical significance (Mean +/- Std).
% 5. Cumulative Route Selection: Starting from 2 seed routes up to 8 routes.
% =========================================================================

function [alResultsTable, fig] = run_route_active_learning(numRandomTrials, saveOutputs)
    if nargin < 1 || isempty(numRandomTrials)
        numRandomTrials = 20; % 20 repeated random trials for statistical rigor
    end
    if nargin < 2 || isempty(saveOutputs)
        saveOutputs = true;
    end

    % Load complete unsplit dataset
    [~, ~, ~, ~, allData] = load_urban_drive_test_data([], 'none');

    testRouteIDs = [10, 11, 12];
    testSet = allData(ismember(allData.RouteID, testRouteIDs), :);

    candidateRouteIDs = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    candidateData = allData(ismember(allData.RouteID, candidateRouteIDs), :);

    routeBudget = 2:8;
    numSteps = length(routeBudget);

    rmse_random_runs = zeros(numRandomTrials, numSteps);
    mae_random_runs  = zeros(numRandomTrials, numSteps);

    fprintf('======================================================================\n');
    fprintf('  RUNNING ROUTE-AWARE ACTIVE LEARNING OPTIMIZATION (%d TRIALS)\n', numRandomTrials);
    fprintf('======================================================================\n');

    % ---------------------------------------------------------------------
    % 1. Passive Random Route Selection (20 Repeated Independent Trials)
    % ---------------------------------------------------------------------
    % For random-trial RMSE benchmarking we need statistically sound RMSE
    % distributions, NOT per-subset uncertainty estimates. Therefore we
    % pre-fit ARD hyperparameters on all candidate data ONCE (using the
    % same 2-feature Spatial ARD GPR [X, Y] that is the proposed model
    % throughout this project) and reuse those fixed hyperparameters for
    % all 140 random-trial GPR evaluations.
    % This eliminates 139 redundant ARD optimizations with no methodological
    % cost: the hyperparameters represent the full candidate route distribution
    % and are applied symmetrically across all random subsets.
    fprintf('  Pre-fitting ARD hyperparameters on full candidate pool (Spatial GPR [X,Y], used for random trials)...\n');
    gpr_full = physics_informed_gpr_model(candidateData, {'X', 'Y'});
    fixed_ell = gpr_full.ell_vec;   % [ell_X, ell_Y]  (2-element vector)
    fixed_sf  = gpr_full.sigma_f;
    fixed_sn  = gpr_full.sigma_n;
    fprintf('  Fixed hyperparams for random trials: ell=[%s], sf=%.3f, sn=%.3f\n', ...
        num2str(fixed_ell, '%.3f '), fixed_sf, fixed_sn);

    for trial = 1:numRandomTrials
        rng(100 + trial, 'twister');
        shuffledRoutes = candidateRouteIDs(randperm(length(candidateRouteIDs)));

        for s = 1:numSteps
            budget = routeBudget(s);
            selectedRoutes = shuffledRoutes(1:budget);

            trData = candidateData(ismember(candidateData.RouteID, selectedRoutes), :);
            % Use pre-fitted ARD hyperparams (fixed, 2-feature [X,Y]) — appropriate for RMSE benchmarking.
            % Passing the full ell_vec (both dimensions) preserves the ARD structure of the proposed model.
            gpr = physics_informed_gpr_model(trData, {'X', 'Y'}, ...
                fixed_ell, fixed_sf, fixed_sn);  % full 2-D ARD mode with pre-fitted hyperparams

            [predMean, ~] = gpr.predict(testSet);
            m = evaluate_predictions(testSet.RSRP, predMean);
            rmse_random_runs(trial, s) = m.RMSE;
            mae_random_runs(trial, s)  = m.MAE;
        end
    end

    % ---------------------------------------------------------------------
    % 2. Uncertainty-Guided Active Route Selection (Cumulative)

    % ---------------------------------------------------------------------
    rmse_active = zeros(numSteps, 1);
    mae_active  = zeros(numSteps, 1);
    selectedActiveRoutes = cell(numSteps, 1);

    % Start deterministically with initial seed routes [1, 2]
    activeRoutes = [1, 2];
    selectedActiveRoutes{1} = activeRoutes;

    % Initial evaluation at budget = 2 (proposed model: Spatial ARD GPR [X, Y])
    trData = candidateData(ismember(candidateData.RouteID, activeRoutes), :);
    gpr = physics_informed_gpr_model(trData, {'X', 'Y'});
    [predMean, ~] = gpr.predict(testSet);
    m = evaluate_predictions(testSet.RSRP, predMean);
    rmse_active(1) = m.RMSE;
    mae_active(1)  = m.MAE;

    for s = 2:numSteps
        targetBudget = routeBudget(s);

        while length(activeRoutes) < targetBudget
            % Fit Spatial ARD GPR [X, Y] (proposed model) on current active training routes
            trData = candidateData(ismember(candidateData.RouteID, activeRoutes), :);
            gpr = physics_informed_gpr_model(trData, {'X', 'Y'});

            % Evaluate latent uncertainty on remaining unvisited candidate routes
            unvisitedRoutes = setdiff(candidateRouteIDs, activeRoutes);
            routeUncertainties = zeros(length(unvisitedRoutes), 1);

            for rIdx = 1:length(unvisitedRoutes)
                rID = unvisitedRoutes(rIdx);
                rPoints = candidateData(candidateData.RouteID == rID, :);
                [~, sigma_latent] = gpr.predict(rPoints);
                routeUncertainties(rIdx) = mean(sigma_latent); % Average latent route uncertainty
            end

            % Select the route with the highest latent model uncertainty
            [maxUncertainty, bestIdx] = max(routeUncertainties);
            bestRoute = unvisitedRoutes(bestIdx);
            activeRoutes = [activeRoutes, bestRoute];
            fprintf('  -> Budget %d: Selected Route %d (Mean Uncertainty = %.2f dB)\n', ...
                length(activeRoutes), bestRoute, maxUncertainty);
        end

        selectedActiveRoutes{s} = activeRoutes;

        % Evaluate on Fixed Unseen Test Routes (proposed model: Spatial ARD GPR [X, Y])
        trData = candidateData(ismember(candidateData.RouteID, activeRoutes), :);
        gpr = physics_informed_gpr_model(trData, {'X', 'Y'});
        [predMean, ~] = gpr.predict(testSet);
        m = evaluate_predictions(testSet.RSRP, predMean);
        rmse_active(s) = m.RMSE;
        mae_active(s)  = m.MAE;
    end

    % Compute deterministic selection order (just routes in activeRoutes at max budget)
    finalActiveOrder = selectedActiveRoutes{end};  % Full ordered sequence at max budget

    % Summary statistics for random selection
    mean_rmse_rand = mean(rmse_random_runs, 1)';

    std_rmse_rand  = std(rmse_random_runs, [], 1)';
    ci95_rmse_rand = 1.96 * std_rmse_rand / sqrt(numRandomTrials);

    % Construct summary table
    alResultsTable = table(routeBudget', mean_rmse_rand, std_rmse_rand, ci95_rmse_rand, rmse_active, mae_active, ...
        'VariableNames', {'MeasuredRoutesBudget', 'Random_MeanRMSE_dB', 'Random_StdRMSE_dB', 'Random_CI95_dB', ...
                          'Active_RMSE_dB', 'Active_MAE_dB'});

    disp(alResultsTable);

    % ---------------------------------------------------------------------
    % Target-Accuracy Efficiency Analysis: Routes Needed for RMSE < 3 dB
    % (Replaces the misleading "35% savings" claim — now a proper analysis)
    % ---------------------------------------------------------------------
    targetRMSE = 3.0;
    fprintf('\n  === TARGET-ACCURACY EFFICIENCY ANALYSIS (RMSE < %.1f dB) ===\n', targetRMSE);

    activeHit = find(rmse_active < targetRMSE, 1, 'first');
    if ~isempty(activeHit)
        nActive = routeBudget(activeHit);
        fprintf('  Active Learning first hits RMSE < %.1f dB at: %d routes\n', targetRMSE, nActive);
    else
        nActive = Inf;
        fprintf('  Active Learning: never achieves RMSE < %.1f dB within budget.\n', targetRMSE);
    end

    randomHit = find(mean_rmse_rand < targetRMSE, 1, 'first');
    if ~isempty(randomHit)
        nRandom = routeBudget(randomHit);
        fprintf('  Random Selection first hits RMSE < %.1f dB at: %d routes\n', targetRMSE, nRandom);
    else
        nRandom = Inf;
        fprintf('  Random Selection: never achieves RMSE < %.1f dB within budget.\n', targetRMSE);
    end

    if isfinite(nActive) && isfinite(nRandom) && nActive < nRandom
        savings = (nRandom - nActive) / nRandom * 100;
        fprintf('  => Active Learning achieves target with %.0f fewer routes (%.1f%% measurement savings)\n\n', ...
            nRandom - nActive, savings);
    elseif isfinite(nActive) && isfinite(nRandom)
        fprintf('  => Both strategies reach target at same budget (%d routes). No savings.\n\n', nActive);
    else
        fprintf('  => Cannot compute savings (one or both strategies never reached target).\n\n');
    end

    % ---------------------------------------------------------------------
    % Plot Publication-Quality Active Learning Comparison
    % ---------------------------------------------------------------------
    fig = figure('Name', 'Route-Aware Active Learning Optimization', 'Color', 'w', 'Position', [120, 120, 920, 520]);
    
    % Random sampling confidence band & line
    fill([routeBudget, fliplr(routeBudget)], ...
         [mean_rmse_rand - std_rmse_rand; flipud(mean_rmse_rand + std_rmse_rand)]', ...
         [1.0 0.85 0.85], 'EdgeColor', 'none', 'DisplayName', 'Random Route Selection (\pm 1\sigma Band)');
    hold on;
    plot(routeBudget, mean_rmse_rand, '-or', 'LineWidth', 2.2, 'MarkerSize', 8, ...
         'DisplayName', sprintf('Passive Random Selection (Mean of %d Trials)', numRandomTrials));

    % Active selection curve
    plot(routeBudget, rmse_active, '-s', 'Color', [0.05 0.55 0.25], 'LineWidth', 2.6, 'MarkerSize', 9, ...
         'MarkerFaceColor', [0.05 0.55 0.25], 'DisplayName', 'Uncertainty-Guided Active Route Selection (\sigma_{latent})');

    xlabel('Number of Measured Candidate Routes (Drive-Test Budget)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    ylabel('Held-Out Test RMSE on Unseen Routes (dB)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    title('Route-Aware Active Learning: Generalization vs. Measurement Budget', ...
        'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');
    legend('Location', 'northeast', 'TextColor', 'k', 'FontSize', 10);
    grid on; box on; set(gca, 'XColor', 'k', 'YColor', 'k');
    xlim([1.8, 8.2]);

    if saveOutputs
        if ~exist('results', 'dir'), mkdir('results'); end
        if ~exist(fullfile('results', 'figures'), 'dir'), mkdir(fullfile('results', 'figures')); end

        writetable(alResultsTable, fullfile('results', 'active_learning_results.csv'));
        saveas(fig, fullfile('results', 'figures', 'active_learning_comparison.png'));
        saveas(fig, fullfile('results', 'active_learning_comparison.png')); % Root copy

        % -----------------------------------------------------------------
        % Report Key Efficiency Claim at Maximum Budget (7 candidate routes)
        % -----------------------------------------------------------------
        budgetIdx7 = find(routeBudget == 7);
        if ~isempty(budgetIdx7)
            rmse_rand_7  = mean_rmse_rand(budgetIdx7);
            rmse_active_7 = rmse_active(budgetIdx7);
            errReduction = (rmse_rand_7 - rmse_active_7) / rmse_rand_7 * 100;
            fprintf('\n  *** Active vs. Random at 7-Route Budget ***\n');
            fprintf('  Random RMSE  = %.2f dB\n', rmse_rand_7);
            fprintf('  Active RMSE  = %.2f dB\n', rmse_active_7);
            fprintf('  Error Reduction = %.1f%%\n\n', errReduction);
        end

        % -----------------------------------------------------------------
        % Figure 2: Route Selection Map — Which Routes Selected & When?
        % -----------------------------------------------------------------
        figMap = figure('Name', 'Active Learning: Route Selection Map', 'Color', 'w', ...
            'Position', [160, 160, 960, 680]);
        colorOrder = lines(9);  % Distinct color per selection step

        hold on;
        for stepIdx = 1:length(finalActiveOrder)
            rID = finalActiveOrder(stepIdx);
            rPts = candidateData(candidateData.RouteID == rID, :);
            h = plot(rPts.X, rPts.Y, '-', 'Color', colorOrder(stepIdx, :), ...
                'LineWidth', 2.5, 'DisplayName', sprintf('Route %d (Selected #%d)', rID, stepIdx));
        end

        % Overlay unselected candidate routes in gray
        unselectedRoutes = setdiff(candidateRouteIDs, finalActiveOrder);
        for rID = unselectedRoutes
            rPts = candidateData(candidateData.RouteID == rID, :);
            plot(rPts.X, rPts.Y, '-', 'Color', [0.75 0.75 0.75], 'LineWidth', 1.2, ...
                'DisplayName', sprintf('Route %d (Not Selected)', rID));
        end

        % Overlay test routes in dashed black
        for rID = testRouteIDs
            tPts = testSet(testSet.RouteID == rID, :);
            plot(tPts.X, tPts.Y, '--k', 'LineWidth', 2.0, ...
                'DisplayName', sprintf('Test Route %d (Unseen)', rID));
        end

        % Label selection numbers on start-of-route
        for stepIdx = 1:length(finalActiveOrder)
            rID = finalActiveOrder(stepIdx);
            rPts = candidateData(candidateData.RouteID == rID, :);
            text(rPts.X(1), rPts.Y(1), sprintf('%d', stepIdx), ...
                'FontSize', 10, 'FontWeight', 'bold', ...
                'Color', colorOrder(stepIdx, :), 'HorizontalAlignment', 'center');
        end

        xlabel('Local X Coordinate (m)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Local Y Coordinate (m)', 'FontSize', 12, 'FontWeight', 'bold');
        title({'Active Learning Route Selection Map', ...
            'Number = Selection Order (higher uncertainty selected first)'}, ...
            'FontSize', 13, 'FontWeight', 'bold');
        legend('Location', 'bestoutside', 'FontSize', 9);
        grid on; box on; axis equal;

        saveas(figMap, fullfile('results', 'figures', 'active_learning_route_selection.png'));
        saveas(figMap, fullfile('results', 'active_learning_route_selection.png'));
        fprintf('Saved Active Learning results to results/active_learning_results.csv and figures.\n');
    end
end

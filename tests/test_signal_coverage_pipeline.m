classdef test_signal_coverage_pipeline < matlab.unittest.TestCase
    % TEST_SIGNAL_COVERAGE_PIPELINE Automated unit test suite
    % MathWorks Excellence in Innovation - Project #151
    % Fulfills MathWorks Repository Guideline 6 (Testing & Verification)

    properties
        ProjectRoot
        SampleData
        PreTrainedGPR
    end

    methods (TestClassSetup)
        function setupPaths(testCase)
            testCase.ProjectRoot = fileparts(fileparts(mfilename('fullpath')));
            if isempty(testCase.ProjectRoot), testCase.ProjectRoot = pwd; end
            addpath(genpath(fullfile(testCase.ProjectRoot, 'src')));
            
            % Load sample data
            sampleCsv = fullfile(testCase.ProjectRoot, 'data', 'sample', 'sample_urban_grid.csv');
            testCase.verifyTrue(exist(sampleCsv, 'file') == 2, 'Sample dataset must exist');
            [~, ~, ~, ~, testCase.SampleData] = load_urban_drive_test_data(sampleCsv, 'none');

            % Load pre-trained GPR
            modelFile = fullfile(testCase.ProjectRoot, 'models', 'gpr_spatial_model.mat');
            testCase.verifyTrue(exist(modelFile, 'file') == 2, 'Pre-trained GPR model must exist');
            m = load(modelFile);
            testCase.PreTrainedGPR = m.gprModel;
        end
    end

    methods (Test)
        function testDataSchemaAndColumns(testCase)
            data = testCase.SampleData;
            reqVars = {'X', 'Y', 'Distance', 'LogDist', 'Azimuth', 'RSRP'};
            for i = 1:length(reqVars)
                testCase.verifyTrue(ismember(reqVars{i}, data.Properties.VariableNames), ...
                    sprintf('Missing column: %s', reqVars{i}));
            end
            testCase.verifyFalse(any(isnan(data.RSRP)), 'RSRP cannot contain NaN');
        end

        function testSpatialGPRInference(testCase)
            model = testCase.PreTrainedGPR;
            [pred, sigmaLatent] = model.predict(testCase.SampleData);
            
            testCase.verifyEqual(length(pred), height(testCase.SampleData));
            testCase.verifyTrue(all(sigmaLatent > 0), 'Bayesian latent sigma must be strictly positive');
            testCase.verifyFalse(any(isnan(pred)), 'Predictions must not contain NaN');
            
            metrics = evaluate_predictions(testCase.SampleData.RSRP, pred);
            testCase.verifyLessThan(metrics.RMSE, 10.0, 'RMSE on sample data should be under 10 dB');
            testCase.verifyGreaterThan(metrics.R2, 0.5, 'R2 score on sample data should exceed 0.5');
        end

        function testEmpiricalLogDistanceModel(testCase)
            logModel = log_distance_model(testCase.SampleData);
            pred = logModel.predict(testCase.SampleData);
            testCase.verifyEqual(length(pred), height(testCase.SampleData));
            testCase.verifyFalse(any(isnan(pred)), 'Log-distance predictions must not contain NaN');
        end

        function testKnnIdwModel(testCase)
            idwModel = knn_idw_model(testCase.SampleData, 2, 10);
            pred = idwModel.predict(testCase.SampleData);
            testCase.verifyEqual(length(pred), height(testCase.SampleData));
            testCase.verifyFalse(any(isnan(pred)), 'IDW predictions must not contain NaN');
        end

        function testEvaluationMetricsComputation(testCase)
            yTrue = [-70; -80; -90];
            yPred = [-70; -80; -90];
            m = evaluate_predictions(yTrue, yPred);
            testCase.verifyEqual(m.RMSE, 0, 'AbsTol', 1e-6);
            testCase.verifyEqual(m.MAE, 0, 'AbsTol', 1e-6);
            testCase.verifyEqual(m.R2, 1, 'AbsTol', 1e-6);
        end

        function testZeroSpatialLeakageProtocol(testCase)
            fullCsv = fullfile(testCase.ProjectRoot, 'data', 'synthetic_urban_grid_dataset.csv');
            if exist(fullCsv, 'file')
                testRoutes = [10, 11, 12];
                [tr, te] = load_urban_drive_test_data(fullCsv, 'route_holdout', testRoutes);
                overlap = intersect(unique(tr.RouteID), unique(te.RouteID));
                testCase.verifyEmpty(overlap, 'Zero-leakage violation: Route overlap detected between train and test');
            end
        end
    end
end

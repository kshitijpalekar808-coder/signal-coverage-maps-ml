%% RUN ALL UNIT TESTS (MathWorks Challenge #151)
% Executes automated MATLAB Unit Testing Framework suite
% Usage: run_tests

clear; clc;
fprintf('======================================================================\n');
fprintf('  RUNNING MATLAB UNIT TEST SUITE (MathWorks Guideline 6)\n');
fprintf('======================================================================\n\n');

testFolder = fileparts(mfilename('fullpath'));
if isempty(testFolder), testFolder = fullfile(pwd, 'tests'); end

suite = matlab.unittest.TestSuite.fromFile(fullfile(testFolder, 'test_signal_coverage_pipeline.m'));
runner = matlab.unittest.TestRunner.withTextOutput;
results = runner.run(suite);

disp(results);
if all([results.Passed])
    fprintf('\n>>> ALL %d UNIT TESTS PASSED SUCCESSFULLY! <<<\n', length(results));
else
    error('Some unit tests failed.');
end

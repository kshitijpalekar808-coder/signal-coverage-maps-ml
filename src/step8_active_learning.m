%% BACKWARD-COMPATIBILITY ADAPTER: STEP 8 ACTIVE LEARNING
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Bridges legacy calls to step8_active_learning(...) to run_route_active_learning.
% =========================================================================

function [alTable, fig] = step8_active_learning(numTrials, saveFig)
    if nargin < 1 || isempty(numTrials)
        numTrials = 20;
    end
    if nargin < 2 || isempty(saveFig)
        saveFig = true;
    end
    [alTable, fig] = run_route_active_learning(numTrials, saveFig);
end

%% MODEL 1: 3GPP-INSPIRED UMi PATH-LOSS REFERENCE BASELINE
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Description:
% Implements a deterministic 3GPP-inspired urban micro (UMi) path-loss reference:
%   PL(d) = 32.4 + 20*log10(f_GHz) + n*10*log10(d_m)
%
% For synthetic benchmark experiments, n=3.0 represents mean-field urban NLOS.
% For real field data, it provides an uncalibrated physics reference.
%
% This model requires NO training or parameter tuning on measurement data.
% It serves as an untrained deterministic propagation reference.
% =========================================================================

function theoModel = theoretical_propagation_model(bsInfo, P_tx, G_ant)
    if nargin < 1 || isempty(bsInfo)
        bsInfo = struct();
    end

    if isstruct(bsInfo)
        if isfield(bsInfo, 'txPower_dBm'), P_tx = bsInfo.txPower_dBm;
        elseif isfield(bsInfo, 'TxPower_dBm'), P_tx = bsInfo.TxPower_dBm;
        else, P_tx = 43; end

        if isfield(bsInfo, 'antennaGain_dBi'), G_ant = bsInfo.antennaGain_dBi;
        elseif isfield(bsInfo, 'AntGain_dBi'), G_ant = bsInfo.AntGain_dBi;
        else, G_ant = 15; end

        if isfield(bsInfo, 'freq_GHz'), f_GHz = bsInfo.freq_GHz;
        elseif isfield(bsInfo, 'Freq_MHz'), f_GHz = bsInfo.Freq_MHz / 1000;
        else, f_GHz = 2.1; end
    else
        % Caller passed scalar freq
        f_GHz = bsInfo;
        if f_GHz > 100, f_GHz = f_GHz / 1000; end % MHz to GHz
        if nargin < 2 || isempty(P_tx), P_tx = 43; end
        if nargin < 3 || isempty(G_ant), G_ant = 15; end
    end

    theoModel.predict = @(inputData) predictTheoretical(inputData, P_tx, G_ant, f_GHz);
    theoModel.name = '3GPP-Inspired UMi Path-Loss Reference (No Training)';
end

function pred_rsrp = predictTheoretical(inputData, P_tx, G_ant, f_GHz)
    if istable(inputData)
        if ismember('Distance', inputData.Properties.VariableNames)
            d = inputData.Distance;
        else
            d = sqrt(inputData.X.^2 + inputData.Y.^2) + 1;
        end
    else
        d = inputData;
    end
    
    d = max(1, d); % Minimum distance 1 meter

    % 3GPP-Inspired Urban Path Loss
    pathLoss = 32.4 + 20 * log10(f_GHz) + 30.0 * log10(d);

    % Received RSRP = P_tx + G_ant - PathLoss
    pred_rsrp = P_tx + G_ant - pathLoss;
end

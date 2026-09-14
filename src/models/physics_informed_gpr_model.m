% Gaussian Process Regression with Anisotropic ARD Matérn 5/2 Kernel
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% Author: Kshitij Palekar
%
% Formulation:
%   r_ARD^2(x, x') = sum_j (x_j - x'_j)^2 / ell_j^2
%   k(x, x') = sigma_f^2 * (1 + sqrt(5)*r + 5*r^2/3) * exp(-sqrt(5)*r)
%
% Hyperparameter Optimization:
%   Learned by maximizing Marginal Log-Likelihood (MLL) on training data
%   via Nelder-Mead simplex search (fminsearch).
%
% Latent Spatial Field Uncertainty:
%   sigma_latent^2(x*) = k(x*, x*) - K_*^T (K + sigma_n^2*I)^-1 K_*

function gprModel = physics_informed_gpr_model(trainData, featureNames, l, sigma_f, sigma_n)
    if nargin < 2 || isempty(featureNames)
        featureNames = {'X', 'Y', 'Distance', 'LogDist', 'Azimuth'};
    end

    X_train = table2array(trainData(:, featureNames));
    y_train = trainData.RSRP;
    p = length(featureNames);

    % =========================================================
    % Training-Only Standardization (Zero Data Leakage)
    % =========================================================
    mu_X = mean(X_train, 1);
    std_X = std(X_train, [], 1);
    std_X(std_X == 0) = 1;

    mu_y = mean(y_train);
    std_y = std(y_train);
    if std_y == 0, std_y = 1; end

    X_tr_norm = (X_train - mu_X) ./ std_X;
    y_tr_norm = (y_train - mu_y) / std_y;
    N = size(X_tr_norm, 1);

    % =========================================================
    % Hyperparameter Initialization & Optimization (Toolbox-Free)
    % =========================================================
    % Check if caller supplied fixed hyperparameters (for unit tests / legacy)
    if nargin >= 3 && ~isempty(l) && nargin >= 4 && ~isempty(sigma_f) && nargin >= 5 && ~isempty(sigma_n)
        % Fixed hyperparameters mode (pre-fitted hyperparams reuse, or unit test).
        % l may be a scalar (isotropic) or a p-element vector (full ARD).
        if isscalar(l)
            ell_vec = l * ones(1, p);   % expand scalar to isotropic
        else
            assert(numel(l) == p, ...
                'physics_informed_gpr_model: length-scale vector length (%d) must match number of features (%d).', ...
                numel(l), p);
            ell_vec = l(:)';            % accept pre-fitted ARD vector as-is
        end
        sf      = sigma_f;
        sn      = sigma_n;
        useARD  = false;
    else
        % -------------------------------------------------------
        % ARD mode: optimize all hyperparameters on training data
        % using log-reparameterization so params stay positive.
        %
        % Log-parameterization (no Optimization Toolbox needed):
        %   phi_j = log(ell_j)  =>  ell_j = exp(phi_j) > 0  always
        %   phi_{p+1} = log(sigma_f),  phi_{p+2} = log(sigma_n)
        %
        % This converts the bounded problem to an unconstrained one
        % solvable by fminsearch (Nelder-Mead, base MATLAB).
        % -------------------------------------------------------
        useARD = true;

        % Initial log-params: ell=1.0, sigma_f=1.0, sigma_n=0.1
        phi0 = [zeros(1, p), 0.0, log(0.1)];   % log([1,...,1, 1.0, 0.1])

        opts = optimset('Display', 'off', 'MaxIter', 400, 'TolFun', 1e-6, 'TolX', 1e-6);

        % For large training sets (N > 1000), subsample up to 1000 points
        % to evaluate MLL efficiently during optimization (standard GPR practice),
        % while building the full prior covariance matrix over all N samples.
        if N > 1000
            rng_mll = rng;
            rng(42, 'twister');
            sub_mll = randperm(N, 1000);
            rng(rng_mll);
            X_mll = X_tr_norm(sub_mll, :);
            y_mll = y_tr_norm(sub_mll);
            N_mll = 1000;
        else
            X_mll = X_tr_norm;
            y_mll = y_tr_norm;
            N_mll = N;
        end

        phi_opt = fminsearch(@(phi) negMarginalLogLik_log(phi, X_mll, y_mll, N_mll, p), ...
            phi0, opts);

        % Recover positive hyperparameters via exp
        params  = exp(phi_opt);
        ell_vec = params(1:p);
        sf      = params(p + 1);
        sn      = params(p + 2);

        % Clamp to sensible range to prevent degenerate solutions
        ell_vec = max(0.01, min(20.0, ell_vec));
        sf      = max(0.05, min(10.0, sf));
        sn      = max(0.005, min(2.0, sn));

        fprintf('  ARD GPR Learned Hyperparameters (MLL-optimized on training set):\n');
        for j = 1:p
            fprintf('    %-12s: ell = %.4f\n', featureNames{j}, ell_vec(j));
        end
        fprintf('  sigma_f = %.4f,  sigma_n = %.4f\n', sf, sn);
    end


    % =========================================================
    % Build Prior Covariance Matrix K (ARD Matérn 5/2)
    % =========================================================
    D2_tr = ardDist2(X_tr_norm, X_tr_norm, ell_vec);
    D_tr  = sqrt(max(0, D2_tr));
    sqrt5_d = sqrt(5) * D_tr;
    K = sf^2 * (1 + sqrt5_d + (5 * D2_tr) / 3) .* exp(-sqrt5_d);

    % Numerically stable Cholesky with adaptive jitter
    jitter = 1e-6;
    while true
        K_noise = K + (sn^2 + jitter) * eye(N);
        [L, flag] = chol(K_noise, 'lower');
        if flag == 0, break; end
        jitter = jitter * 10;
        if jitter > 1e-1
            error('Physics-Informed GPR: Covariance matrix not positive-definite even with jitter=%.1e', jitter);
        end
    end

    % Dual vector alpha = K_noise^-1 y
    alpha = L' \ (L \ y_tr_norm);

    % =========================================================
    % Package Trained Model Struct
    % =========================================================
    gprModel.name           = 'Physics-Informed ARD GPR (Matérn 5/2)';
    gprModel.featureNames   = featureNames;
    gprModel.useARD         = useARD;
    gprModel.mu_X           = mu_X;
    gprModel.std_X          = std_X;
    gprModel.mu_y           = mu_y;
    gprModel.std_y          = std_y;
    gprModel.X_tr_norm      = X_tr_norm;
    gprModel.alpha          = alpha;
    gprModel.L              = L;
    gprModel.ell_vec        = ell_vec;
    gprModel.sigma_f        = sf;
    gprModel.sigma_n        = sn;
    gprModel.jitter         = jitter;

    if useARD
        gprModel.lengthScaleTable = cell2table([featureNames', num2cell(ell_vec')], ...
            'VariableNames', {'Feature', 'LearnedLengthScale'});
    end

    gprModel.predict = @(inputData) predictARD_GPR(gprModel, inputData);
end

% =========================================================================
% ARD GPR Prediction (Mean + Latent Uncertainty + Observation Uncertainty)
% =========================================================================
function [mu, sigma_latent, sigma_obs] = predictARD_GPR(gpr, inputData)
    if istable(inputData)
        X_eval = table2array(inputData(:, gpr.featureNames));
    else
        X_eval = inputData;
    end

    X_te_norm = (X_eval - gpr.mu_X) ./ gpr.std_X;
    M = size(X_te_norm, 1);
    sf2 = gpr.sigma_f^2;

    mu          = zeros(M, 1);
    sigma_latent = zeros(M, 1);
    sigma_obs   = zeros(M, 1);

    chunkSize = 2000;
    for startIdx = 1:chunkSize:M
        endIdx = min(startIdx + chunkSize - 1, M);
        sub_Xte = X_te_norm(startIdx:endIdx, :);

        D2_te = ardDist2(sub_Xte, gpr.X_tr_norm, gpr.ell_vec);
        D_te  = sqrt(max(0, D2_te));
        sqrt5_te = sqrt(5) * D_te;
        K_star = sf2 * (1 + sqrt5_te + (5 * D2_te) / 3) .* exp(-sqrt5_te);

        % Predictive mean
        mu_norm = K_star * gpr.alpha;
        mu(startIdx:endIdx) = mu_norm * gpr.std_y + gpr.mu_y;

        % Latent variance: sigma_latent^2 = k(x*,x*) - v^T v,  where L*v = K_*^T
        v = gpr.L \ K_star';   % [N x chunk]
        var_reduction = sum(v.^2, 1)';
        var_latent_norm = max(0, sf2 - var_reduction);
        var_latent_dB = var_latent_norm * (gpr.std_y^2);
        sigma_latent(startIdx:endIdx) = sqrt(var_latent_dB);

        % Observation variance = latent + noise
        sigma_obs(startIdx:endIdx) = sqrt(var_latent_dB + (gpr.sigma_n * gpr.std_y)^2);
    end
end

% =========================================================================
% ARD Squared Distance: r^2_j = (x_j - x'_j)^2 / ell_j^2
% =========================================================================
function D2 = ardDist2(A, B, ell_vec)
    % A: [M x p], B: [N x p], ell_vec: [1 x p]
    A_scaled = bsxfun(@rdivide, A, ell_vec);  % [M x p]
    B_scaled = bsxfun(@rdivide, B, ell_vec);  % [N x p]

    sumA = sum(A_scaled.^2, 2);   % [M x 1]
    sumB = sum(B_scaled.^2, 2);   % [N x 1]
    D2 = bsxfun(@plus, sumA, sumB') - 2 * (A_scaled * B_scaled');
    D2 = max(0, D2);
end

% =========================================================================
% Negative Marginal Log-Likelihood (Log-Parameterized, Toolbox-Free)
% phi = log([ell_1,...,ell_p, sigma_f, sigma_n])  (unconstrained in R^{p+2})
% Params recovered via exp(phi) — always strictly positive.
% =========================================================================
function nmll = negMarginalLogLik_log(phi, X_norm, y_norm, N, p)
    params  = exp(phi);
    ell_vec = params(1:p);
    sf      = params(p + 1);
    sn      = params(p + 2);

    % Safety clamp: prevent extreme values from crashing Cholesky
    ell_vec = max(1e-3, min(50.0, ell_vec));
    sf      = max(1e-3, min(20.0, sf));
    sn      = max(1e-4, min(5.0,  sn));

    D2 = ardDist2(X_norm, X_norm, ell_vec);
    D  = sqrt(max(0, D2));
    sqrt5d = sqrt(5) * D;
    K = sf^2 * (1 + sqrt5d + (5 * D2) / 3) .* exp(-sqrt5d);

    K_noise = K + (sn^2 + 1e-6) * eye(N);
    [L, flag] = chol(K_noise, 'lower');
    if flag ~= 0
        nmll = 1e10;
        return;
    end

    alpha = L' \ (L \ y_norm);

    % NMLL = 0.5*y^T*alpha + sum(log diag(L)) + N/2*log(2*pi)
    data_fit   = 0.5 * (y_norm' * alpha);
    complexity = sum(log(diag(L)));
    nmll = data_fit + complexity + 0.5 * N * log(2 * pi);
end

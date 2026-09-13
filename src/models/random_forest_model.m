% Random Forest Regression Ensemble for Signal Coverage Mapping
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% Author: Kshitij Palekar
%
% Implements regression tree ensemble with Out-of-Bag (OOB) feature importance
% and automatic pure MATLAB fallback for portability across environments.

function rfModel = random_forest_model(trainData, featureNames, numTrees, randomSeed)
    if nargin < 2 || isempty(featureNames)
        featureNames = {'X', 'Y', 'Distance', 'LogDist', 'Azimuth'};
    end
    if nargin < 3 || isempty(numTrees)
        numTrees = 150;
    end
    if nargin < 4 || isempty(randomSeed)
        randomSeed = 42;
    end

    rng(randomSeed, 'twister');

    X_train = trainData(:, featureNames);
    y_train = trainData.RSRP;

    % Check if Statistics and Machine Learning Toolbox TreeBagger exists
    hasTreeBagger = (exist('TreeBagger', 'file') == 2 || exist('TreeBagger', 'class') == 8);

    if hasTreeBagger
        try
            tb = TreeBagger(numTrees, X_train, y_train, ...
                'Method', 'regression', ...
                'MinLeafSize', 5, ...
                'OOBPredictorImportance', 'on');

            rfModel.tb = tb;
            rfModel.featureNames = featureNames;
            rfModel.OOBPermutedPredictorImportance = tb.OOBPermutedPredictorImportance;
            rfModel.name = sprintf('Random Forest (%d Trees - TreeBagger)', numTrees);
            rfModel.predict = @(inputData) predictTreeBagger(tb, inputData, featureNames);
            return;
        catch
            % Fall back to high-performance native engine
        end
    end

    % High-performance pure MATLAB Random Forest engine
    X_mat = table2array(X_train);
    y_vec = y_train;
    rfNative = trainNativeRF(X_mat, y_vec, numTrees, 5, featureNames, randomSeed);

    rfModel.native = rfNative;
    rfModel.featureNames = featureNames;
    rfModel.OOBPermutedPredictorImportance = rfNative.OOBImportance;
    rfModel.name = sprintf('Random Forest (%d Trees - Native MATLAB)', numTrees);
    rfModel.predict = @(inputData) predictNativeRF(rfNative, inputData, featureNames);
end

function y_pred = predictTreeBagger(tb, inputData, featureNames)
    if istable(inputData)
        X_eval = inputData(:, featureNames);
    else
        X_eval = array2table(inputData, 'VariableNames', featureNames);
    end
    y_pred = predict(tb, X_eval);
    if iscell(y_pred), y_pred = cellfun(@str2double, y_pred); end
end

function rf = trainNativeRF(X, y, numTrees, minLeaf, featureNames, seed)
    rng(seed, 'twister');
    [n, p] = size(X);
    numSubspace = max(1, floor(p / 2));

    trees = cell(numTrees, 1);
    oobIndices = cell(numTrees, 1);

    for b = 1:numTrees
        bootIdx = randi(n, n, 1);
        oob = setdiff(1:n, unique(bootIdx));
        trees{b} = buildNode(X(bootIdx, :), y(bootIdx), minLeaf, numSubspace, 0, 15);
        oobIndices{b} = oob;
    end

    % Compute OOB permutation feature importance
    baselineOOBErrors = zeros(n, 1);
    baselineCounts = zeros(n, 1);
    for b = 1:numTrees
        oob = oobIndices{b};
        if isempty(oob), continue; end
        preds = predictSingleTree(trees{b}, X(oob, :));
        baselineOOBErrors(oob) = baselineOOBErrors(oob) + (preds - y(oob)).^2;
        baselineCounts(oob) = baselineCounts(oob) + 1;
    end
    vIdx = baselineCounts > 0;
    baseMSE = mean(baselineOOBErrors(vIdx) ./ baselineCounts(vIdx));

    oobImportance = zeros(1, p);
    for j = 1:p
        permErrors = zeros(n, 1);
        permCounts = zeros(n, 1);
        for b = 1:numTrees
            oob = oobIndices{b};
            if isempty(oob), continue; end
            X_perm = X(oob, :);
            X_perm(:, j) = X_perm(randperm(length(oob)), j);
            preds = predictSingleTree(trees{b}, X_perm);
            permErrors(oob) = permErrors(oob) + (preds - y(oob)).^2;
            permCounts(oob) = permCounts(oob) + 1;
        end
        vIdx = permCounts > 0;
        permMSE = mean(permErrors(vIdx) ./ permCounts(vIdx));
        oobImportance(j) = max(0.01, permMSE - baseMSE);
    end

    rf.trees = trees;
    rf.numTrees = numTrees;
    rf.featureNames = featureNames;
    rf.OOBImportance = oobImportance;
end

function node = buildNode(X, y, minLeaf, numSubspace, depth, maxDepth)
    n = size(X, 1);
    if n <= minLeaf || depth >= maxDepth || var(y) < 1e-4
        node.isLeaf = true;
        node.prediction = mean(y);
        return;
    end

    p = size(X, 2);
    candFeats = randperm(p, min(p, numSubspace));
    bestCost = Inf;
    bestF = 0;
    bestTh = 0;

    for f = candFeats
        vals = sort(unique(X(:, f)));
        if length(vals) <= 1, continue; end
        if length(vals) > 20
            q = linspace(5, 95, 15) / 100;
            ths = vals(max(1, round(q * length(vals))));
        else
            ths = (vals(1:end-1) + vals(2:end)) / 2;
        end

        for th = ths(:)'
            left = X(:, f) <= th;
            right = ~left;
            if sum(left) < minLeaf || sum(right) < minLeaf, continue; end
            cost = sum((y(left) - mean(y(left))).^2) + sum((y(right) - mean(y(right))).^2);
            if cost < bestCost
                bestCost = cost;
                bestF = f;
                bestTh = th;
            end
        end
    end

    if bestF == 0
        node.isLeaf = true;
        node.prediction = mean(y);
        return;
    end

    node.isLeaf = false;
    node.feature = bestF;
    node.threshold = bestTh;
    leftMask = X(:, bestF) <= bestTh;
    node.left = buildNode(X(leftMask, :), y(leftMask), minLeaf, numSubspace, depth + 1, maxDepth);
    node.right = buildNode(X(~leftMask, :), y(~leftMask), minLeaf, numSubspace, depth + 1, maxDepth);
end

function preds = predictNativeRF(rf, inputData, featureNames)
    if istable(inputData)
        X_mat = table2array(inputData(:, featureNames));
    else
        X_mat = inputData;
    end

    M = size(X_mat, 1);
    treePreds = zeros(M, rf.numTrees);
    for b = 1:rf.numTrees
        treePreds(:, b) = predictSingleTree(rf.trees{b}, X_mat);
    end
    preds = mean(treePreds, 2);
end

function y_hat = predictSingleTree(tree, X)
    M = size(X, 1);
    y_hat = zeros(M, 1);
    for i = 1:M
        cur = tree;
        while ~cur.isLeaf
            if X(i, cur.feature) <= cur.threshold
                cur = cur.left;
            else
                cur = cur.right;
            end
        end
        y_hat(i) = cur.prediction;
    end
end

%% UTILITY: VECTORIZED PAIRWISE EUCLIDEAN DISTANCE COMPUTATION
% =========================================================================
% Project: Signal Coverage Maps Using Measurements and Machine Learning
% MathWorks Excellence in Innovation - Project #151
% =========================================================================
% Function: compute_pairwise_dist
% Computes the Euclidean distance matrix between rows of matrix A (M x P)
% and matrix B (N x P) using numerically stable vectorized expansion.
% Output: D (M x N) where D(i,j) = ||A(i,:) - B(j,:)||_2
% =========================================================================

function D = compute_pairwise_dist(A, B)
    if nargin < 2 || isempty(B)
        B = A;
    end
    
    sumA = sum(A.^2, 2);      % [M x 1]
    sumB = sum(B.^2, 2);      % [N x 1]
    
    % D^2 = sumA + sumB' - 2 * A * B'
    D2 = bsxfun(@plus, sumA, sumB') - 2 * (A * B');
    
    % Clamp negative values arising from floating-point roundoff to zero
    D2(D2 < 0) = 0;
    D = sqrt(D2);
end

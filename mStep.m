function [newC] = mStep(Y, mu, cov)

[T, ~] = size(Y);
[~, L] = size(mu);

% Accumulate cov across all time points into 'S'
% TODO account for missing data in S
cov_parts = mat2cell(cov, T * ones(1, L), T * ones(1, L));
cov_parts_tt = cellfun(@diag, cov_parts, 'UniformOutput', false);
S = cellfun(@sum, cov_parts_tt);

% Set nan values to zero
Y(isnan(Y)) = 0;

% Solve for new C that maximizes the Q function of EM
newC = (Y' * mu) / (S + mu' * mu);

end
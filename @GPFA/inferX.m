function [mu_x, sigma_x] = inferX(gpfaObj, queryTimes)

assert(isempty(gpfaObj.Sf), 'Sf must be empty to infer X alone, otherwise use GPFA.inferMeanFieldXF');

if ~exist('queryTimes', 'var') || isempty(queryTimes), queryTimes = gpfaObj.times; end

L = gpfaObj.L;

if length(queryTimes) == length(gpfaObj.times) && all(queryTimes == gpfaObj.times)
    sigma_x = gpfaObj.Cov;
    queryIdx = 1:gpfaObj.T;
    nPad = 0;
    T = gpfaObj.T;
else
    allTimes = [gpfaObj.times setdiff(queryTimes, gpfaObj.times)];
    T = length(allTimes);
    [~, queryIdx] = ismember(queryTimes, allTimes);
    nPad = T - gpfaObj.T;
    gpfaObj.dt = [];
    gpfaObj.times = allTimes;
    gpfaObj = gpfaObj.updateK();
    % The following is a copy of gpfaObj.updateCov
    K = gpfaObj.K;
    GammaCells = mat2cell(gpfaObj.Gamma, gpfaObj.T*ones(1, L), gpfaObj.T*ones(1, L));
    G = cell2mat(cellfun(@(Gam) spblkdiag(Gam, sparse(nPad, nPad)), GammaCells, 'UniformOutput', false));
    I = speye(size(G));
    blocks = T * ones(1, L);
    sigma_x = K - K * G * blockmldivide((I + K * G), blocks, K);
end

% TODO - faster implementation when there is no missing data ?

residual = gpfaObj.Y - gpfaObj.b';
if ~isempty(gpfaObj.S)
    residual = residual - gpfaObj.S * gpfaObj.D';
end
residual(isnan(residual)) = 0;
residual = vertcat(residual, zeros(nPad, gpfaObj.N));

RiC = gpfaObj.C ./ gpfaObj.R;

mu_x = sigma_x * reshape(residual * RiC, T * L, 1);
mu_x = reshape(mu_x, T, L);

mu_x = mu_x(queryIdx, :);
sigma_subs = queryIdx(:) + T*(0:L-1);
sigma_x = sigma_x(sigma_subs(:), sigma_subs(:));

end
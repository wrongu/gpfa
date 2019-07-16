function [mu_x, sigma_x] = inferX(gpfaObj, queryTimes)

assert(isempty(gpfaObj.Sf), 'Sf must be empty to infer X alone, otherwise use GPFA.inferMeanFieldXF');
            
if ~gpfaObj.initialized, gpfaObj = gpfaObj.updateAll(); end

if ~exist('queryTimes', 'var') || isempty(queryTimes), queryTimes = gpfaObj.times; end

L = gpfaObj.L;

if L == 0
    mu_x = [];
    sigma_x = {};
    return
end

if length(queryTimes) == length(gpfaObj.times) && all(queryTimes == gpfaObj.times)
    sigma_x = gpfaObj.Cov;
    Gamma = gpfaObj.Gamma;
    queryIdx = 1:gpfaObj.T;
    nPad = 0;
    newT = gpfaObj.T;
else
    % 'allTimes' is the set of original time points followed by disjoint queried times. It is
    % deliberately not in order.
    allTimes = [gpfaObj.times setdiff(queryTimes, gpfaObj.times)];
    newT = length(allTimes);
    [~, queryIdx] = ismember(queryTimes, allTimes);
    nPad = newT - gpfaObj.T;
    gpfaObj.dt = [];
    gpfaObj.times = allTimes;
    gpfaObj = gpfaObj.updateK();
    Gamma = cellfun(@(G) padarray(G, [nPad nPad], 0, 'post'), gpfaObj.Gamma, 'UniformOutput', false);
    % The following is a copy of gpfaObj.updateCov
    for l=L:-1:1
        K = gpfaObj.K{l};
        G = Gamma{l,l};
        I = speye(size(G));
        sigma_x{l} = K - K * G * ((I + K * G) \ K);
    end
end

% TODO - faster implementation when there is no missing data ?

residual = gpfaObj.Y - gpfaObj.b';
if ~isempty(gpfaObj.S)
    residual = residual - gpfaObj.S * gpfaObj.D';
end
residual(isnan(residual)) = 0;
residual = vertcat(residual, zeros(nPad, gpfaObj.N));

RiC = gpfaObj.C ./ gpfaObj.R;

mu_x = zeros(newT, L);
if L > 1
    % Explaining-away must be handled iteratively due to factorized posterior approximation
    for itr=1:10
        for l=1:L
            l_other = [1:l-1 l+1:L];
            proj_x_other = zeros(newT, 1);
            for l2=l_other
                proj_x_other = proj_x_other + Gamma{l, l2} * mu_x(:, l2);
            end
            mu_x(:, l) = gather(sigma_x{l} * (residual * RiC(:, l) - proj_x_other));
        end
    end
else
    mu_x = gather(sigma_x{1} * residual * RiC);
end

mu_x = mu_x(queryIdx, :);
sigma_x = cellfun(@(sig) gather(sig(queryIdx, queryIdx)), sigma_x, 'UniformOutput', false);

end
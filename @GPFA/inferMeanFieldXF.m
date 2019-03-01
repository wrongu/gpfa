function [mu_x, sigma_x, mu_f, sigma_f] = inferMeanFieldXF(gpfaObj, queryTimes, queryStims, iters)

assert(~isempty(gpfaObj.Sf), 'Sf is empty - use inferX instead!');

if ~exist('queryTimes', 'var') || isempty(queryTimes), queryTimes = gpfaObj.times; end
if ~exist('queryStims', 'var') || isempty(queryStims), queryStims = gpfaObj.uSf; end
if ~exist('iters', 'var') || isempty(iters), iters=5; end

L = gpfaObj.L;
N = gpfaObj.N;
RiC = gpfaObj.C ./ gpfaObj.R;

% Handle 'query' time indices
if length(queryTimes) == length(gpfaObj.times) && all(queryTimes == gpfaObj.times)
    % Assuming updateCov() was called recently, nothing else to be done
    sigma_x = gpfaObj.Cov;
    Gamma = gpfaObj.Gamma;
    queryTimeIdx = 1:gpfaObj.T;
    newT = gpfaObj.T;
    nTimePad = 0;
else
    % 'allTimes' is the set of original time points followed by disjoint queried times. It is
    % deliberately not in order.
    allTimes = [gpfaObj.times setdiff(queryTimes, gpfaObj.times)];
    newT = length(allTimes);
    [~, queryTimeIdx] = ismember(queryTimes, allTimes);
    nTimePad = newT - gpfaObj.T;
    gpfaObj.dt = [];
    gpfaObj.times = allTimes;
    gpfaObj = gpfaObj.updateK();
    Gamma = cellfun(@(G) spblkdiag(G, sparse(nTimePad, nTimePad)), gpfaObj.Gamma, 'UniformOutput', false);
    % The following is a copy of gpfaObj.updateCov
    % The following is a copy of gpfaObj.updateCov
    for l=L:-1:1
        K = gpfaObj.K{l};
        G = Gamma{l,l};
        I = speye(size(G));
        sigma_x{l} = K - K * G * ((I + K * G) \ K);
    end
end
baseTimeIdx = 1:gpfaObj.T;

% Handle 'query' tuning curve indices
if all(size(queryStims) == size(gpfaObj.uSf)) && all(queryStims(:) == gpfaObj.uSf(:))
    queryStimIdx = 1:size(gpfaObj.uSf, 1);
    nStimPad = 0;
    
    % Compute sigma_f using standard method
    for n=N:-1:1
        K = gpfaObj.signs(n)^2 * gpfaObj.Kf;
        G = spdiag(gpfaObj.Ns) / gpfaObj.R(n);
        
        % The following is equivalent to inv(inv(K) + G) but doesn't require taking inv(K) directly
        sigma_f{n} = K - K * G * ((eye(size(K)) + K * G) \ K);
    end
else
    allStims = [gpfaObj.uSf; setdiff(queryStims, gpfaObj.uSf, 'rows')];
    [~, queryStimIdx] = ismember(queryStims, allStims, 'rows');
    nStimPad = size(allStims, 1) - size(gpfaObj.uSf, 1);
    
    % Compute a new 'ss2', filling in the bottom, right, and corner for 'new' stimulus pairs
    ss2 = blkdiag(gpfaObj.ss2, zeros(nStimPad, nStimPad));
    for iF=1:size(ss2,1)
        for jF=size(gpfaObj.uSf,1)+1:size(ss2, 1)
            ss2(iF, jF) = gpfaObj.stim_dist_fun(allStims(iF, :), allStims(jF, :))^2;
            ss2(jF, iF) = ss2(iF, jF);
        end
    end
    
    % Keep this in sync with GPFA.updateKernelF
    Kf = exp(-ss2 / gpfaObj.tauf^2) + (1e-6)*eye(size(ss2));
    
    % Compute sigma_f using 'padded' Gammas
    for n=N:-1:1
        K = gpfaObj.signs(n)^2 * Kf;
        G = spblkdiag(spdiag(gpfaObj.Ns) / gpfaObj.R(n), sparse(nStimPad, nStimPad));
        
        % The following is equivalent to inv(inv(K) + G) but doesn't require taking inv(K) directly
        sigma_f{n} = K - K * G * ((eye(size(K)) + K * G) \ K);
    end
end
baseStimIdx = 1:size(gpfaObj.uSf, 1);

% residual is [T x N] and contains Y with the baseline and linear stimulus terms subtracted out
residual = gpfaObj.Y - gpfaObj.b';
if ~isempty(gpfaObj.S)
    residual = residual - gpfaObj.S * gpfaObj.D';
end
residual(isnan(residual)) = 0;

    function mu_x = updateX(mu_f)
        % First, expand mu_f to be [T x N]
        mu_f_expanded = mu_f(gpfaObj.Sf_ord, :);
        
        residualF = vertcat(residual - mu_f_expanded, zeros(nTimePad, gpfaObj.N));
        
        mu_x = zeros(newT, L);
        if L > 1
            % Explaining-away must be handled iteratively due to factorized posterior approximation
            for xitr=1:10
                for l1=1:L
                    l_other = [1:l1-1 l1+1:L];
                    proj_x_other = zeros(newT, 1);
                    for l2=l_other
                        proj_x_other = proj_x_other + Gamma{l1, l2} * mu_x(:, l2);
                    end
                    mu_x(:, l1) = sigma_x{l1} * (residualF * RiC(:, l1) - proj_x_other);
                end
            end
        else
            mu_x = sigma_x{1} * residualF * RiC;
        end
    end

    function mu_f = updateF(mu_x)
        residual_with_x = residual - mu_x * gpfaObj.C';
        % Currently 'residual' is [T x N] but we need [S x N] version where each row is the sum of
        % all trials (T) where the stimulus had a particular value (S)
        residualS = zeros(length(gpfaObj.Ns), N);
        for iStim=1:length(gpfaObj.Ns)
            residualS(iStim, :) = sum(residual_with_x(gpfaObj.Sf_ord == iStim, :), 1);
        end
        residualS = residualS ./ gpfaObj.R';
        residualS = vertcat(residualS, zeros(nStimPad, N));
        
        for nn=N:-1:1
            mu_f(:, nn) = sigma_f{nn} * residualS(:, nn);
        end
    end

% Initialize with zero latents (simply fit tuning to start), then do a series of coordinate-ascent
% updates, ultimately converging to the factorized q(x)q(f) which best approximates p(x,f|...)
mu_x = zeros(newT, L);
for itr=1:iters
    mu_f = updateF(mu_x(baseTimeIdx, :));
    mu_x = updateX(mu_f(baseStimIdx, :));
end

% Subselect to get 'queried' points
mu_x = mu_x(queryTimeIdx, :);
sigma_x = cellfun(@(sig) sig(queryTimeIdx, queryTimeIdx), sigma_x, 'UniformOutput', false);

mu_f = mu_f(queryStimIdx, :);
sigma_f = cellfun(@(sig) sig(queryStimIdx, queryStimIdx), sigma_f, 'UniformOutput', false);

end
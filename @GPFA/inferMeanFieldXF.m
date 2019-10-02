function [mu_x, sigma_x, mu_f, sigma_f] = inferMeanFieldXF(gpfaObj, queryTimes, queryStims, maxIters, convTol)

assert(~isempty(gpfaObj.Sf), 'Sf is empty - use inferX instead!');

if ~gpfaObj.initialized, gpfaObj = gpfaObj.updateAll(); end

if ~exist('queryTimes', 'var') || isempty(queryTimes), queryTimes = gpfaObj.times; end
if ~exist('queryStims', 'var') || isempty(queryStims), queryStims = gpfaObj.uSf; end
if ~exist('iters', 'var') || isempty(maxIters), maxIters=500; end
if ~exist('convTol', 'var') || isempty(convTol), convTol=1e-6; end

for k=gpfaObj.nGP:-1:1
    if gpfaObj.forceZeroF(k)
        % Ensure there is a zero point queried
        augmentedQueryStims{k} = unique(vertcat(queryStims{k}, zeros(1, size(gpfaObj.Sf{k}, 2))), 'rows');
    else
        augmentedQueryStims{k} = unique(queryStims{k}, 'rows');
    end
end

L = gpfaObj.L;
N = gpfaObj.N;

if L > 0
    RiC = gpfaObj.C ./ gpfaObj.R;
else
    RiC = 0;
end

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
    Gamma = cellfun(@(G) padarray(G, [nTimePad nTimePad], 0, 'post'), gpfaObj.Gamma, 'UniformOutput', false);
    
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
for k=gpfaObj.nGP:-1:1
    if all(size(augmentedQueryStims{k}) == size(gpfaObj.uSf{k})) && all(augmentedQueryStims{k}(:) == gpfaObj.uSf{k}(:))
        queryStimIdx{k} = 1:size(gpfaObj.uSf{k}, 1);
        nStimPad(k) = 0;
        
        % Compute sigma_f using standard method
        for n=N:-1:1
            K = gpfaObj.signs(k,n)^2 * gpfaObj.Kf{k};
            G = spdiag(gpfaObj.Ns{k}(n,:)) / gpfaObj.R(n);
            
            % The following is equivalent to inv(inv(K) + G) but doesn't require taking inv(K) directly
            sigma_f{k}{n} = K - K * G * ((eye(size(K)) + K * G) \ K);
        end
        newF(k) = size(gpfaObj.Ns{k}, 2);
        
        if gpfaObj.forceZeroF(k)
            idxZeroStim(k) = find(all(gpfaObj.uSf{k} == 0, 2));
        end
    else
        allStims = [gpfaObj.uSf{k}; setdiff(augmentedQueryStims{k}, gpfaObj.uSf{k}, 'rows')];
        [~, queryStimIdx{k}] = ismember(queryStims{k}, allStims, 'rows');
        nStimPad(k) = size(allStims, 1) - size(gpfaObj.uSf{k}, 1);
        
        % Compute a new 'ss2', filling in the bottom, right, and corner for 'new' stimulus pairs
        ss = blkdiag(sqrt(gpfaObj.ss2{k}), zeros(nStimPad(k), nStimPad(k)));
        for iF=1:size(ss,1)
            for jF=size(gpfaObj.uSf{k},1)+1:size(ss, 1)
                ss(iF, jF) = gpfaObj.stim_dist_fun{k}(allStims(iF, :), allStims(jF, :));
                ss(jF, iF) = ss(iF, jF);
            end
        end
        
        % Keep this in sync with GPFA.updateKernelF
        Kf = exp(-ss.^2 / gpfaObj.tauf(k)^2) + 1e-6 * eye(size(ss));
        
        % Compute sigma_f using 'padded' Gammas
        for n=N:-1:1
            K = gpfaObj.signs(k,n)^2 * Kf;
            G = padarray(spdiag(gpfaObj.Ns{k}(n,:)) / gpfaObj.R(n), [nStimPad(k) nStimPad(k)], 0, 'post');
            
            % The following is equivalent to inv(inv(K) + G) but doesn't require taking inv(K) directly
            sigma_f{k}{n} = K - K * G * ((eye(size(K)) + K * G) \ K);
        end
        newF(k) = size(allStims, 1);
        
        if gpfaObj.forceZeroF(k)
            idxZeroStim(k) = find(all(allStims == 0, 2));
        end
    end
    baseStimIdx{k} = 1:size(gpfaObj.uSf{k}, 1);
end

% residual is [T x N] and contains Y with the baseline and linear stimulus terms subtracted out
residual = gpfaObj.Y - gpfaObj.b';
if ~isempty(gpfaObj.S)
    residual = residual - gpfaObj.S * gpfaObj.D';
end
missing_data = isnan(residual);

    function mu_x = updateX(mu_f, last_mu_x)
        residualF = residual;
        for kk=1:gpfaObj.nGP
            % First, expand mu_f to be [T x N]
            mu_f_expanded = mu_f{kk}(gpfaObj.Sf_ord{kk}, :);
            % Subtract of all prediction from kth stimulus tuning term
            residualF = residualF - mu_f_expanded;
        end
        residualF(missing_data) = 0;
        
        % Pad zeros onto residuals for inference over 'query' time points
        residualF = vertcat(residualF, zeros(nTimePad, gpfaObj.N));
        
        if L == 0
            mu_x = [];
            return
        elseif L > 1
            mu_x = last_mu_x;
            % Explaining-away across xs must be handled iteratively due to factorized posterior
            % approximation
            delta_x = inf;
            while delta_x / numel(mu_x) > convTol
                for l1=1:L
                    l_other = [1:l1-1 l1+1:L];
                    proj_x_other = zeros(newT, 1);
                    for l2=l_other
                        proj_x_other = proj_x_other + Gamma{l1, l2} * mu_x(:, l2);
                    end
                    mu_x(:, l1) = gather(sigma_x{l1} * (residualF * RiC(:, l1) - proj_x_other));
                end
                delta_x = max(abs(mu_x - last_mu_x));
                last_mu_x = mu_x;
            end
        else
            mu_x = gather(sigma_x{1} * residualF * RiC);
        end
    end

    function mu_f = updateF(mu_x, last_mu_f)
        if L > 0
            residual_with_x = residual - mu_x * gpfaObj.C';
        else
            residual_with_x = residual;
        end
        residual_with_x(missing_data) = 0;
        
        % Get prediction for *each* GP stimulus term, to be updated iteratively below.
        mu_f = last_mu_f;
        pred_f = cell(size(last_mu_f));
        for kk=1:gpfaObj.nGP
            pred_f{kk} = mu_f{kk}(gpfaObj.Sf_ord{kk}, :);
        end
        
        % Explaining-away across different fs must be handled iteratively due to factorized
        % posterior approximation. Kick-start the iteration by initializing all mu_f to wherever we
        % left off in the previous outer iteration.
        if gpfaObj.nGP == 1
            % Currently 'residual' is [T x N] but we need [S x N] version where each row is the sum
            % of all trials (T) where the stimulus had a particular value (S)
            residualS = zeros(size(gpfaObj.Ns{1}, 2), N);
            for iStim=1:size(gpfaObj.Ns{1}, 2)
                residualS(iStim, :) = sum(residual_with_x(gpfaObj.Sf_ord{1} == iStim, :), 1);
            end
            residualS = residualS ./ gpfaObj.R';
            residualS = vertcat(residualS, zeros(nStimPad, N));

            for nn=N:-1:1
                mu_f{1}(:, nn) = sigma_f{1}{nn} * residualS(:, nn);
            end
            
            if gpfaObj.forceZeroF
                mu_f{1} = mu_f{1} - mu_f{1}(idxZeroStim, :);
            end
        else
            delta_f = inf;
            last_mu_f = cellcat(last_mu_f);
            while delta_f / numel(last_mu_f) > convTol
                for kk=1:gpfaObj.nGP
                    k_other = [1:kk-1 kk+1:gpfaObj.nGP];
                    pred_k_other = sum(cat(3, pred_f{k_other}), 3);
                    full_residual = residual_with_x - pred_k_other;
                    full_residual(missing_data) = 0;
                    % Currently 'residual' is [T x N] but we need [S x N] version where each row is the
                    % sum of all trials (T) where the stimulus had a particular value (S)
                    residualS = zeros(size(gpfaObj.Ns{kk}, 2), N);
                    for iStim=1:size(gpfaObj.Ns{kk}, 2)
                        residualS(iStim, :) = sum(full_residual(gpfaObj.Sf_ord{kk} == iStim, :), 1);
                    end
                    residualS = residualS ./ gpfaObj.R';
                    residualSPadded = vertcat(residualS, zeros(nStimPad(kk), N));
                    
                    for nn=N:-1:1
                        mu_f{kk}(:, nn) = sigma_f{kk}{nn} * residualSPadded(:, nn);
                    end
            
                    if gpfaObj.forceZeroF(kk)
                        mu_f{kk} = mu_f{kk} - mu_f{kk}(idxZeroStim(kk), :);
                    end
                end
                delta_f = max(abs(cellcat(mu_f) - last_mu_f));
                last_mu_f = cellcat(mu_f);
            end
        end
    end

if L == 0
    % Special case... it's ugly, but at least indexing an empty array with an empty array doesn't
    % throw an error.
    baseTimeIdx = [];
    queryTimeIdx = [];
    sigma_x = {};
end

% Initialize with zero latents (simply fit tuning to start), then do a series of coordinate-ascent
% updates, ultimately converging to the factorized q(x)q(f) which best approximates p(x,f|...)
mu_x = zeros(newT, L);
for k=gpfaObj.nGP:-1:1
    mu_f{k} = zeros(newF(k), N);
end
% Iterate to convergence or max iters
itr = 2;
delta = inf;
while delta(itr-1) > convTol && itr <= maxIters
    new_mu_f = updateF(mu_x(baseTimeIdx, :), mu_f);
    new_base_f = arrayfun(@(k) new_mu_f{k}(baseStimIdx{k}, :), 1:gpfaObj.nGP, 'UniformOutput', false);
    new_mu_x = updateX(new_base_f, mu_x);
    delta(itr) = max([abs(cellcat(mu_f) - cellcat(new_mu_f)); abs(mu_x(:) - new_mu_x(:))]);
    mu_f = new_mu_f;
    mu_x = new_mu_x;
    itr = itr + 1;
end

% Subselect to get 'queried' points
mu_x = mu_x(queryTimeIdx, :);
sigma_x = cellfun(@(sig) gather(sig(queryTimeIdx, queryTimeIdx)), sigma_x, 'UniformOutput', false);

for k=1:gpfaObj.nGP
    mu_f{k} = mu_f{k}(queryStimIdx{k}, :);
    sigma_f{k} = cellfun(@(sig) sig(queryStimIdx{k}, queryStimIdx{k}), sigma_f{k}, 'UniformOutput', false);
end

end

function v = cellcat(c)
v = zeros(sum(cellfun(@numel, c)), 1);
j = 1;
for i=1:length(c)
    v(j:j+numel(c{i})-1) = c{i}(:);
    j = j+numel(c{i});
end
end
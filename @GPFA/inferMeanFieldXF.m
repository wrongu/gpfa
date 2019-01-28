function [mu_x, sigma_x, mu_f, sigma_f] = inferMeanFieldXF(gpfaObj, Y, iters)

assert(~isempty(gpfaObj.Sf), 'Sf is empty - use inferX instead!');

if ~exist('iters', 'var'), iters=5; end

if ~exist('Y', 'var')
    Y = gpfaObj.Y;
else
    gpfaObj.Y = Y;
    gpfaObj = gpfaObj.updateGamma(Y);
    gpfaObj = gpfaObj.updateCov();
end

% residual is [T x N] and contains Y with the baseline and linear stimulus terms subtracted out
residual = Y - gpfaObj.b';
if ~isempty(gpfaObj.S)
    residual = residual - gpfaObj.S * gpfaObj.D';
end
residual(isnan(residual)) = 0;

% Note that covariances can be computed once - the dependence of X on F and vice versa is only
% through their means.

% Assuming updateCov() was called recently, nothing else to be done
sigma_x = gpfaObj.Cov;

for n=gpfaObj.N:-1:1
    K = gpfaObj.signs(n)^2 * gpfaObj.Kf;
    G = spdiag(gpfaObj.Ns) / gpfaObj.R(n);

    % The following is equivalent to inv(inv(K) + G) but doesn't require taking inv(K) directly
    sigma_f{n} = K - K * G * ((eye(size(K)) + K * G) \ K);
end

    function mu_x = updateX(mu_f)
        % First, expand mu_f to be [T x N]
        mu_f_expanded = mu_f(gpfaObj.Sf_ord, :);
        
        RiC = gpfaObj.C ./ gpfaObj.R;
        mu_x = sigma_x * flatten((residual - mu_f_expanded) * RiC);
        mu_x = reshape(mu_x, gpfaObj.T, gpfaObj.L);
    end

    function mu_f = updateF(mu_x)
        residual_with_x = residual - mu_x * gpfaObj.C';
        % Currently 'residual' is [T x N] but we need [S x N] version where each row is the sum of
        % all trials (T) where the stimulus had a particular value (S)
        residualS = zeros(length(gpfaObj.Ns), gpfaObj.N);
        for iStim=1:length(gpfaObj.Ns)
            residualS(iStim, :) = sum(residual_with_x(gpfaObj.Sf_ord == iStim, :), 1);
        end
        residualS = residualS ./ gpfaObj.R';
        
        for nn=gpfaObj.N:-1:1
            mu_f(:, nn) = sigma_f{nn} * residualS(:, nn);
        end
    end

% Initialize with zero latents (simply fit tuning to start), then do a series of coordinate-ascent
% updates, ultimately converging to the factorized q(x)q(f) which best approximates p(x,f|...)
mu_x = 0;
sigma_x = gpfaObj.Cov;
for itr=1:iters
    mu_f = updateF(mu_x);
    mu_x = updateX(mu_f);
end
end

function v = flatten(A)
v = A(:);
end
function [gpfaObj, Q] = emStep(gpfaObj, fixedParams)
R = gpfaObj.R;
T = gpfaObj.T;
L = gpfaObj.L;
C = gpfaObj.C;
Y = gpfaObj.Y;
b = gpfaObj.b;
Gamma = gpfaObj.Gamma;

if nargin < 2, fixedParams = {}; end

if ~isempty(gpfaObj.S)
    stim_predict = gpfaObj.S * gpfaObj.D';
else
    stim_predict = 0;
end

%% E-Step

[mu_x, sigma_x] = gpfaObj.inferX();

% sigma_tt will contain values of sigma_x at all t1==t2
sigma_tt = zeros(L, L, T);
for t=1:T
    sigma_tt(:, :, t) = sigma_x(t:T:T*L, t:T:T*L);
end
% Compute E[x'x] from sigma_tt
expected_xx = (mu_x' * mu_x) + sum(sigma_tt, 3);

y_resid = Y - b' - stim_predict;
y_resid(isnan(y_resid)) = 0;
y_resid_Ri = y_resid ./ R';

logdet_R = sum(log(R));
trace_gamma_sigma = sum(sum(Gamma' .* sigma_x));
neg_2Q = T * logdet_R + sum(sum(y_resid .* y_resid_Ri)) - 2 * sum(sum(y_resid_Ri .* (mu_x * C'))) ...
    + vec(mu_x)' * Gamma * vec(mu_x) + trace_gamma_sigma;
Q = -1/2 * neg_2Q;

%% M-Step

if ~any(strcmp('R', fixedParams))
    residual = (Y - b' - mu_x * C' - stim_predict);
    residual(isnan(residual)) = 0;
    gpfaObj.R = diag((residual' * residual + C * sum(sigma_tt, 3) * C') / T);
end

if ~any(strcmp('b', fixedParams))
    residual = Y - mu_x * C' - stim_predict;
    gpfaObj.b = nanmean(residual, 1)';
end

if ~any(strcmp('C', fixedParams))
    residual = Y - b' - stim_predict;
    residual(isnan(residual)) = 0;
    gpfaObj.C = (residual' * mu_x) / expected_xx;
end

if ~any(strcmp('D', fixedParams)) && ~isempty(gpfaObj.S)
    residual = Y - b' - mu_x * C';
    residual(isnan(residual)) = 0;
    gpfaObj.D = residual' * gpfaObj.S / (gpfaObj.S' * gpfaObj.S);
end

%% Update precomputed matrices
gpfaObj = gpfaObj.updateAll();

end

function x = vec(x)
x = x(:);
end
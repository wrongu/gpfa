function [Yhat, x] = simulate(gpfaObj)
%GPFA.SIMULATE generate data sampled from the prior. Result is [T x N] matrix of Y data.

% Draw values of x from the prior
mu_x = zeros(gpfaObj.T * gpfaObj.L, 1);
x = reshape(mvnrnd(mu_x, gpfaObj.K), gpfaObj.T, gpfaObj.L);

mu_Y = gpfaObj.b' + x * gpfaObj.C';

if ~isempty(gpfaObj.S)
    mu_Y = mu_Y + gpfaObj.S * gpfaObj.D';
end

noise_Y = randn(gpfaObj.T, gpfaObj.N) .* sqrt(gpfaObj.R)';

Yhat = mu_Y + noise_Y;

end
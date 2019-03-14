function [Yhat, x, f] = simulate(gpfaObj)
%GPFA.SIMULATE generate data sampled from the prior. Result is [T x N] matrix of Y data.

if gpfaObj.L > 0
    % Draw values of x from the prior
    x = zeros(gpfaObj.T, gpfaObj.L);
    mu_x_l = zeros(gpfaObj.T, 1);
    for l=1:gpfaObj.L
        x(:, l) = mvnrnd(mu_x_l, gpfaObj.K{l});
    end
    
    mu_Y = gpfaObj.b' + x * gpfaObj.C';
else
    x = [];
    mu_Y = repmat(gpfaObj.b', gpfaObj.T, 1);
end

if ~isempty(gpfaObj.S)
    mu_Y = mu_Y + gpfaObj.S * gpfaObj.D';
end

if ~isempty(gpfaObj.Sf)
    % Draw values of f from the prior
    dimf = size(gpfaObj.Kf, 1);
    f = gpfaObj.signs .* real(sqrtm(gpfaObj.Kf) * randn(dimf, gpfaObj.N));
    mu_Y = mu_Y + f(gpfaObj.Sf_ord, :);
else
    f = [];
end

noise_Y = randn(gpfaObj.T, gpfaObj.N) .* sqrt(gpfaObj.R)';

Yhat = mu_Y + noise_Y;

end
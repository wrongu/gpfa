function [mu_x, sigma_x] = inferX(gpfaObj, Y)

if ~exist('Y', 'var')
    Y = gpfaObj.Y;
else
    gpfaObj.Y = Y;
    gpfaObj = gpfaObj.updateGamma(Y);
    gpfaObj = gpfaObj.updateCov();
end

% TODO - faster implementation when there is no missing data ?

residual = Y - gpfaObj.b';
if ~isempty(gpfaObj.S)
    residual = residual - gpfaObj.S * gpfaObj.D';
end
residual(isnan(residual)) = 0;

RiC = gpfaObj.C ./ gpfaObj.R;

sigma_x = gpfaObj.Cov;
mu_x = sigma_x * reshape(residual * RiC, gpfaObj.T * gpfaObj.L, 1);
mu_x = reshape(mu_x, gpfaObj.T, gpfaObj.L);

end
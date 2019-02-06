function [mu_x, sigma_x] = inferX(gpfaObj)

assert(isempty(gpfaObj.Sf), 'Sf must be empty to infer X alone, otherwise use GPFA.inferMeanFieldXF');

% TODO - faster implementation when there is no missing data ?

residual = gpfaObj.Y - gpfaObj.b';
if ~isempty(gpfaObj.S)
    residual = residual - gpfaObj.S * gpfaObj.D';
end
residual(isnan(residual)) = 0;

RiC = gpfaObj.C ./ gpfaObj.R;

sigma_x = gpfaObj.Cov;
mu_x = sigma_x * reshape(residual * RiC, gpfaObj.T * gpfaObj.L, 1);
mu_x = reshape(mu_x, gpfaObj.T, gpfaObj.L);

end
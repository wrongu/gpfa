function [Y] = sampleY(gpfaObj, nSamples, mu_x, sigma_x)

if ~exist('mu_x', 'var')
    [mu_x, sigma_x] = gpfaObj.inferX();
end

% Get [TL x nSamples] instances of x. It helps to avoid errors if we doubly ensure that sigma_x is
% symmetric.
x_samples = mvnrnd(mu_x(:)', (sigma_x + sigma_x')/2, nSamples);
% Reshape to [T x L x nSamples]
x_samples = reshape(x_samples, gpfaObj.T, gpfaObj.L, nSamples);

% Dot product of L dimension of x with L dimension of C to get [T x N x nSamples] values for the
% component of Y driven by x.
xC = squeeze(sum(reshape(gpfaObj.C, [1 gpfaObj.N gpfaObj.L 1]) .* reshape(x_samples, [gpfaObj.T 1 gpfaObj.L nSamples]), 3));

Y = gpfaObj.b' + xC;

if ~isempty(gpfaObj.S)
    Y = Y + gpfaObj.S * gpfaObj.D';
end

end
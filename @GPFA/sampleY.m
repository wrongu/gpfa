function [Y] = sampleY(gpfaObj, nSamples, mu_x, sigma_x, mu_f, sigma_f)

if isempty(gpfaObj.Sf)
    if ~exist('mu_x', 'var')
        [mu_x, sigma_x] = gpfaObj.inferX();
    end
else
    if ~exist('mu_x', 'var') || ~exist('mu_f', 'var')
        [mu_x, sigma_x, mu_f, sigma_f] = gpfaObj.inferMeanFieldXF();
    end
end

% Get [T x L x nSamples] instances of x. It helps to avoid errors if we doubly ensure that sigma_x
% is symmetric.
for l=gpfaObj.L:-1:1
    x_samples(:, l, :) = mvnrnd(mu_x(:)', (sigma_x{l} + sigma_x{l}')/2, nSamples);
end

f_samples = zeros(gpfaObj.T, gpfaObj.N, nSamples);
if ~isempty(gpfaObj.Sf)
    % Sample possible tuning curves
    S = size(gpfaObj.Kf, 1);
    f_samples = mymvnrnd(mu_f(:)', full(spblkdiag(sigma_f{:})), nSamples);
    f_samples = reshape(f_samples, S, gpfaObj.N, nSamples);
    % For each sampled tuning curve, repeat it as many times as it appears per session (convert from
    % size [S x N x nSamples] to [T x N x nSamples]
    f_samples = f_samples(gpfaObj.Sf_ord, :, :);
end

% Dot product of L dimension of x with L dimension of C to get [T x N x nSamples] values for the
% component of Y driven by x.
xC = squeeze(sum(reshape(gpfaObj.C, [1 gpfaObj.N gpfaObj.L 1]) .* reshape(x_samples, [gpfaObj.T 1 gpfaObj.L nSamples]), 3));

Y = gpfaObj.b' + xC + f_samples;

if ~isempty(gpfaObj.S)
    Y = Y + gpfaObj.S * gpfaObj.D';
end

end
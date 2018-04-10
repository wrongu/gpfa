function [mu, cov, G, P] = inferX(Y, diagD, C, Kfull)

Yz = Y;
Yz(isnan(Y)) = 0;

[T, ~] = size(Y);
[~, L] = size(C);

% Gamma defines cov across latents at each time point. Gamma is constructed by repeating CDC (the
% cov across latents) 'T' times by multiplying with eye(T) then block-concatenating each eye term
% together.
G = getGammaWithMissingData(Y, diagD, C);

% P is the projection of the data onto the latents [Y*inv(D)*C] with missing data taken into
% account.
P = Yz * (C ./ diagD(:));
P = P(:);

% Posterior mean and precision matrix
precision = (inv(Kfull) + G);
mu = precision \ P;

% Posterior covariance matrix
cov = inv(precision);
% ensure symmetry
cov = (cov + cov') / 2;

mu = reshape(mu, T, L);

end
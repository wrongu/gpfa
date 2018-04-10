function [C, Q, Kfull] = learnLoadingsEM(times, y, sigma_ys, sigma_fs, sigma_xs, taus, latents)
%% Check inputs

[T, N] = size(y);
sigma_ys = sigma_ys(:);
epsilon = 1e-5;

%% Construct GP kernel
for l=latents:-1:1
K{l} = sigma_xs(l)^2 * exp(-(times' - times).^2 / taus(l)) + sigma_fs(l)^2 * eye(T);
end

Kfull = blkdiag(K{:});

%% Run EM

max_iters = 10;

C = ones(N, latents) + randn(N, latents) * .1;
Q = zeros(1, max_iters);
Q(1) = logLikelihoodEstimate(y, sigma_ys, C, Kfull);

for i=1:max_iters
    %% E Step
    [mu, cov] = inferX(y, sigma_ys, C, Kfull);
    
    %% M Step
    C = mStep(y, mu, cov);
    
    %% Check convergence
    Q(i+1) = logLikelihoodEstimate(y, sigma_ys, C, Kfull);
    if abs(Q(i+1)-Q(i)) < epsilon, break; end
    
    fprintf('Iteration %05d:\tQ = %.3e\n', i, Q(i+1));
    subplot(1,2,1);
    cla;
    plot(C);
    subplot(1,2,2);
    cla;
    plot(Q(1:i+1));
    drawnow;
end

% Truncate Q
Q(i+1:end) = [];

end
clear;
rng(12345678, 'twister');

%% Set parameters

T = 20; % time points
N = 10; % data points (neurons)
sigma_data = 0.1; % noise on data
D = sigma_data^2 * eye(N); % noise covariance on data
L = 2; % number of latents
% Each of the following: one for each L
sigma_f = .1 * ones(L, 1);
sigma_x = .5 * ones(L, 1);
tau = 5 * ones(L, 1);

% Fraction of missing (nan) data
datadrop = .3;

%% Generate ground truth

times = linspace(0, 10, T); % Time points equi-spaced

for l=L:-1:1
C(:,l) = sin(linspace(0, 2 * pi, N) + l)'; % Loadings
mu(:,l) = zeros(T, 1);
K{l} = sigma_x(l)^2 * exp(-(times' - times).^2 / tau(l)) + sigma_f(l)^2 * eye(T);
x_true(:,l) = mvnrnd(mu(:,l), K{l});
end

% Data is size [T x N]
data = x_true * C' + mvnrnd(zeros(T, N), D);

data(rand(size(data)) < datadrop) = nan;

%% Infer x

% K defines cov across time for each latent
Kfull = blkdiag(K{:});

[mu_inferred, cov_inferred] = inferX(data, diag(D), C, Kfull);

%% Plot result

% plot 20 random draws from posterior
figure; hold on;
colors = lines(L);
for i=1:20
    x = reshape(mvnrnd(mu_inferred(:), cov_inferred), [T, L]);
    for l=1:L
        plot(times, x(:,l), 'Color', colors(l,:));
    end
end

% Plot ground truth
for l=1:L
    plot(times, x_true(:,l), 'Color', colors(l,:), 'LineWidth', 2);
end

title(sprintf('Inferred vs true latent (%.2f%% missing data)', datadrop*100));
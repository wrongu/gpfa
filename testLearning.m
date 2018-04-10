clear;

%% Set parameters

T = 20; % time points
N = 30; % data points (neurons)
sigma_data = 0.3; % noise on data
D = sigma_data^2 * eye(N); % noise covariance on data
diagD = diag(D);
L = 2; % number of latents
% Each of the following: one for each L
sigma_f = 0 * ones(L, 1);
sigma_x = .5 * ones(L, 1);
tau = 5 * ones(L, 1);

% Fraction of missing (nan) data
datadrop = 0;

%% Generate ground truth

times = linspace(0, 10, T); % Time points equi-spaced

for l=L:-1:1
C_gt(:,l) = sin(linspace(0, 2 * pi, N) + l)'; % Loadings
mu(:,l) = zeros(T, 1);
K{l} = sigma_x(l)^2 * exp(-(times' - times).^2 / tau(l)) + sigma_f(l)^2 * eye(T);
x_true(:,l) = mvnrnd(mu(:,l), K{l});
end

% Data is size [T x N]
data = x_true * C_gt' + mvnrnd(zeros(T, N), D);

data(rand(size(data)) < datadrop) = nan;

%% Run EM

% K defines cov across time for each latent
Kfull = blkdiag(K{:});

% Initialize C to a random guess
C = randn(size(C_gt)) * .1;

iters = 1000;
plotevery = 100;
iplot = 1:plotevery:iters;
Q = zeros(1, iters);

crange = linspace(0, 1, length(iplot))';
colors = [crange zeros(size(crange)) flipud(crange)];
figure;
subplot(1,2,1);
hold on;
plot(C_gt(:), 'Color', 'k', 'LineWidth', 2);
j = 1;
for itr=1:iters
    [mu, cov] = inferX(data, diagD, C, Kfull);
    C = mStep(data, mu, cov);
    if any(iplot == itr)
        h(j) = plot(C(:), 'Color', colors(j, :));
        j = j + 1;
    end
    Q(itr) = logLikelihoodEstimate(data, diagD, C, Kfull);
end
legend(h([1 end]), {'init', 'end'});

subplot(1,2,2);
plot(Q);
xlabel('iteration');
ylabel('Q (log likelihood lower bound');
%% Generate ground truth data

N = 30;
T = 500;
L = 3;
M = 5;

os = ones(1, L);

R = .05 * exp(randn(N, 1) * .2);
taus = gamrnd(2, 10, 1, L);
sigs = .3 * os;
rhos = .001 * os;

b = 10 * rand(N, 1);
C = zeros(N, L);
for l=1:L
    C(:, l) = sin(linspace(0, 2 * pi, N) + l);
end

% Linear 'tuning' to S for each neuron with random slope
sensitivities = randn(N, 1);
D = sensitivities .* linspace(-1, 1, M);

% Construct S as 1-hot per trial
S = zeros(T, M);
for t=T:-1:1
    S(t, randi(M)) = 1;
end

disp('init');
fakeData = zeros(T, N);
gpfa = GPFA(fakeData, L, 'dt', 1, 'R', R, 'taus', taus, 'sigs', sigs, 'rhos', rhos, 'b', b, 'C', C, 'S', S, 'D', D);

disp('generate');
[simData, xTrue] = gpfa.simulate();
gpfa = gpfa.setFields('Y', simData);

figure;
subplot(2,1,1);
imagesc(simData');
xlabel('trials');
ylabel('neurons');
title('data');
colorbar;

stim_pred = S * D';
residuals = simData - stim_pred;
subplot(2,1,2);
imagesc(residuals');
xlabel('trials');
ylabel('neurons');
title('residuals (minus stim)');
colorbar;

%% Test inference

disp('infer');
[mu_x, sigma_x] = gpfa.inferX();

figure;
subplot(2,1,1);
hold on;
colors = lines(L);
for l=1:L
    stdev = sqrt(diag(sigma_x{l}));
    errorbar(mu_x(:, l), sqrt(stdev) / 2, 'Color', colors(l, :));
    plot(xTrue(:, l), 'Color', colors(l, :), 'LineWidth', 2);
end
title('ground truth versus inferred values of latent X');

residuals = simData - gpfa.predictY;
subplot(2,1,2);
imagesc(residuals');
xlabel('trials');
ylabel('neurons');
title('residuals (predict Y with ground truth model)');
colorbar;

%% Test inference at different 'query' times

disp('query');
queryTimes = gpfa.times + .5;
[mu_x_q, sigma_x_q] = gpfa.inferX(queryTimes);

figure;
hold on;
colors = lines(L);
for l=1:L
    stdev = sqrt(diag(sigma_x{l}));
    errorbar(gpfa.times, mu_x(:, l), sqrt(stdev) / 2, 'Color', colors(l, :));
    stdev_q = sqrt(diag(sigma_x_q{l}));
    errorbar(queryTimes, mu_x_q(:, l), sqrt(stdev_q) / 2, 'Color', colors(l, :)/2);
end
title('Inferred X at times vs at interpolated times');

%% Test full learning and monotonic increase of 'Q'

iters = 500;
% init = gpfa.setFields('fixed', {'taus'});
init = GPFA('Y', simData, 'L', L, 'taus', taus, 'rhos', rhos, 'sigs', sigs, 'S', S, 'taus_alpha', 2*os, ...
    'taus_beta', .1*os, 'kernel_update_freq', 10);
[bestFit, Qs] = init.fitEM(iters, 1e-6);

figure;
plot(Qs);
xlabel('iteraion #');
ylabel('Q');

figure;
subplot(2,2,1);
scatter(gpfa.b, bestFit.b);
title('fit vs true offsets ''b''');
axis equal;
subplot(2,2,2); hold on;
for l=1:L
    plot(gpfa.C(:,l), bestFit.C(:,l), 'o', 'Color', colors(l, :));
end
title('fit vs true loadings ''C''');
axis equal;
subplot(2,2,3);
scatter(gpfa.R(:), bestFit.R(:));
title('fit vs true variance ''R''');
axis equal;
subplot(2,2,4);
scatter(gpfa.D(:), bestFit.D(:));
title('fit vs true stim loadings ''D''');
axis equal;

for l=1:gpfa.L
    fprintf('true tau_%d = %f\tfit tau_%d = %f\n', l, gpfa.taus(l), l, bestFit.taus(l));
end

%% Re-test inference using best-fit model

disp('infer');
[mu_x, sigma_x] = bestFit.inferX();

figure;
subplot(2,1,1);
hold on;
colors = lines(L);
for l=1:L
    stdev = sqrt(diag(sigma_x{l}));
    errorbar(mu_x(:, l), sqrt(stdev) / 2, 'Color', colors(l, :));
    plot(xTrue(:, l), 'Color', colors(l, :), 'LineWidth', 2);
end
title('ground truth versus (fit) inferred values of latent X');

residuals = simData - bestFit.predictY;
subplot(2,1,2);
imagesc(residuals');
xlabel('trials');
ylabel('neurons');
title('residuals (predict Y with fitted model)');
colorbar;

%% Try fitting for different sized data subsets, with and without GPU acceleration

fractions = linspace(0.1, 1, 10);
times = 1:T;
iters = 50;

for ifrac=1:length(fractions)
    subset = randperm(T, round(T*fractions(ifrac)));
    
    %% CPU version
    init = GPFA('Y', simData(subset, :), 'L', L, 'times', times(subset), 'taus', taus, 'rhos', rhos, ...
        'sigs', sigs, 'S', S(subset, :), 'taus_alpha', 2*os, 'taus_beta', .1*os, 'kernel_update_freq', 10);
    tstart = tic;
    [~, Q_CPU{ifrac}] = init.fitEM(iters, 1e-6);
    elapsedCPU(ifrac) = toc(tstart);
    
    %% GPU version
    init = init.setFields('useGPU', true);
    tstart = tic;
    [~, Q_GPU{ifrac}] = init.fitEM(iters, 1e-6);
    elapsedGPU(ifrac) = toc(tstart);
end

figure;
subplot(1,2,1);
hold on;
plot(round(fractions*T), elapsedCPU);
plot(round(fractions*T), elapsedGPU);
xlabel('timepoints');
ylabel('fitting time');
legend('CPU', 'GPU');

subplot(1,2,2);
hold on;
colors = lines(length(fractions));
for ifrac=1:length(fractions)
    plot(Q_CPU{ifrac}, '-', 'Color', colors(ifrac,:), 'DisplayName', sprintf('T = %d', round(fractions(ifrac)*T)));
    plot(Q_GPU{ifrac}, 'o', 'Color', colors(ifrac,:), 'DisplayName', sprintf('T = %d', round(fractions(ifrac)*T)));
end
xlabel('iteration');
ylabel('Q');
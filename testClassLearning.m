%% Generate ground truth data

N = 30;
T = 100;
L = 3;
M = 5;

R = .05 * exp(randn(N, 1) * .2);
taus = linspace(5, 30, L);
sigs = .3 * ones(1, L);
rhos = .001 * ones(1, L);

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
gpfa = GPFA(fakeData, L, 'dt', 1, 'R', R, 'taus', taus, 'sigs', sigs, 'rhos', rhos, 'b', b, 'C', C, 'S', S, 'D', D, 'rho_decay', inf);

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
[mu_x, sigma_x] = gpfa.inferX(simData);
variances = diag(sigma_x);

figure;
subplot(2,1,1);
hold on;
colors = lines(L);
for l=1:L
    stdev = sqrt(variances((l-1)*T+1:l*T));
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

%% Test full learning and monotonic increase of 'Q'

iters = 500;
% gpfa = gpfa.setFields('fixed', {'taus'});
init = gpfa.setFields('fixed', {}, 'taus', [10 10 10]);
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
subplot(2,2,2);
scatter(gpfa.C(:), bestFit.C(:));
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
[mu_x, sigma_x, e_xx] = bestFit.inferX(simData);
variances = diag(sigma_x);

figure;
subplot(2,1,1);
hold on;
colors = lines(L);
for l=1:L
    stdev = sqrt(variances((l-1)*T+1:l*T));
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
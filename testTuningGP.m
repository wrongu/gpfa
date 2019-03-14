%% Generate ground truth data

N = 10;
T = 200;
L = 0;
M = 5;
stims = linspace(0, 2*pi*99/100, 100)';
shuffle = randi(length(stims), T, 1);
Sf = stims(shuffle);

R = .05 * exp(randn(N, 1) * .2);
taus = linspace(5, 30, L);
sigs = .3 * ones(1, L);
rhos = exprnd(.01, 1, L);

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
gpfa = GPFA(fakeData, L, 'dt', 1, 'R', R, 'taus', taus, 'sigs', sigs, 'rhos', rhos, 'b', b, 'C', C, ...
    'S', S, 'D', D, 'Sf', Sf, 'tauf', 1, 'signs', logspace(-1,.3,N), 'stim_dist_fun', @circ_dist);

disp('generate');
[simData, xTrue, fTrue] = gpfa.simulate();
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
[mu_x, sigma_x, mu_f, sigma_f] = gpfa.inferMeanFieldXF();

figure;
subplot(3,1,1);
hold on;
colors = lines(L);
for l=1:L
    stdev = sqrt(sqrt(diag(sigma_x{l})));
    errorbar(mu_x(:, l), stdev / 2, 'Color', colors(l, :));
    plot(xTrue(:, l), 'Color', colors(l, :), 'LineWidth', 2);
end
title('ground truth versus inferred values of latent X');

residuals = simData - gpfa.predictY;
subplot(3,1,2);
imagesc(residuals');
xlabel('trials');
ylabel('neurons');
title('residuals (predict Y with ground truth model)');
colorbar;

subplot(3,1,3); hold on;
colors = lines(N);
for n=1:N
    plot(gpfa.uSf, fTrue(:, n), 'LineWidth', 2, 'Color', colors(n, :));
    errorbar(gpfa.uSf, mu_f(:, n), sqrt(diag(sigma_f{n})), 'Color', colors(n, :));
end

%% Test inference at 'queried' points

disp('query');
queryTimes = gpfa.times + 0.5;
queryF = stims;
[mu_x_q, sigma_x_q, mu_f_q, sigma_f_q] = gpfa.inferMeanFieldXF(queryTimes, queryF);

figure;
subplot(2,1,1);
hold on;
colors = lines(L);
for l=1:L
    stdev = sqrt(diag(sigma_x{l}));
    errorbar(gpfa.times, mu_x(:, l), stdev / 2, 'Color', colors(l, :));
    stdev_q = sqrt(diag(sigma_x_q{l}));
    errorbar(queryTimes, mu_x_q(:, l), stdev_q / 2, 'Color', colors(l, :)/2);
end
axis tight;

subplot(2,1,2);
hold on;
colors = lines(N);
for n=1:N
    errorbar(gpfa.uSf, mu_f(:, n), sqrt(diag(sigma_f{n})), '-o', 'Color', colors(n, :));
    errorbar(queryF, mu_f_q(:, n), sqrt(diag(sigma_f_q{n})), 'Color', colors(n, :)/2);
end
axis tight;

%% Test full learning and monotonic increase of 'Q'

iters = 500;
gpfa.lr = 1e-3;
gpfa.fixed = {'signs'};
[bestFit, Qs] = gpfa.fitEM(iters, 1e-6);

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
    fprintf('true rho_%d = %f\tfit rho_%d = %f\n', l, gpfa.rhos(l), l, bestFit.rhos(l));
end

%% Re-test inference using best-fit model

disp('infer');
[mu_x, sigma_x, mu_f, sigma_f] = bestFit.inferMeanFieldXF();

figure;
subplot(3,1,1);
hold on;
colors = lines(L);
for l=1:L
    stdev = sqrt(diag(sigma_x{l}));
    errorbar(mu_x(:, l), stdev / 2, 'Color', colors(l, :));
    plot(xTrue(:, l), 'Color', colors(l, :), 'LineWidth', 2);
end
title('ground truth versus (fit) inferred values of latent X');

residuals = simData - bestFit.predictY;
subplot(3,1,2);
imagesc(residuals');
xlabel('trials');
ylabel('neurons');
title('residuals (predict Y with fitted model)');
colorbar;

subplot(3,1,3); hold on;
colors = lines(N);
for n=1:N
    plot(gpfa.uSf, fTrue(:, n), 'LineWidth', 2, 'Color', colors(n, :));
    errorbar(gpfa.uSf, mu_f(:, n), sqrt(diag(sigma_f{n})), 'Color', colors(n, :));
end
title('tuning curves estimated with fitted model');

%% Helper

function d = circ_dist(a1, a2)
a1 = mod(a1, 2*pi);
a2 = mod(a2, 2*pi);
d = min(abs(a1-a2), 2*pi-abs(a1-a2));
end
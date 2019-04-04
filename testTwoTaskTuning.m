%% Create set of unique 2-task stimuli that share a zero-signal level

Mf = 31;
stimsA = norminv(linspace(.1, .9, Mf));
stimsB = norminv(linspace(.1, .9, Mf));
stims = unique([stimsA(:) zeros(Mf, 1); 0 0; zeros(Mf, 1) stimsB(:)], 'rows');
ns = size(stims, 1);

% Construct the GP covariance kernel
ss = zeros(ns);
for i=1:size(stims,1)
    for j=1:size(stims,1)
        ss(i,j) = twoTaskDistFun(stims(i,:), stims(j,:));
    end
end
kernel = exp(-ss.^2);
kernel = GPFA.fixImpossiblePairwiseCorrs(kernel);

subplot(2,2,1);
imagesc(ss);
axis image;
title('distances');

kernel1 = exp(-ss.^2);
subplot(2,2,2);
imagesc(kernel1);
axis image;
title('naive covariance');

kernel2 = GPFA.fixImpossiblePairwiseCorrs(kernel1);
subplot(2,2,3);
imagesc(kernel2);
axis image;
title('fixed covariance');

kernel3 = GPFA.fixImpossiblePairwiseCorrs(kernel2);
subplot(2,2,4);
imagesc(kernel3);
axis image;
title('2x fixed covariance');

D = eye(size(ss)) / 100;
f1 = real(sqrtm(kernel1 + D) * randn(ns, 1));
f2 = real(sqrtm(kernel2 + D) * randn(ns, 1));
f3 = real(sqrtm(kernel3 + D) * randn(ns, 1));
figure; hold on;
plot(f1);
plot(f2);
plot(f3);

%% Generate ground truth data

N = 3;
T = 200;
L = 1;
M = 2;

% Costruct Sf by randomly selecting rows of 'allStims'
Sf = stims(randi(size(stims, 1), T, 1), :);
uSf = unique(Sf, 'rows');

% Since sampling from GP kernels is hard, we construct 'smooth' f manually
linear_f_slopes = randn(2, N);
linear_f_offsets = rand(1, N) * 5;
f = zeros(size(uSf, 1), N);
for n=1:N
    for i=1:size(f, 1)
        if all(uSf(i, :) == 0)
            f(i, n) = linear_f_offsets(n);
        elseif uSf(i, 1) ~= 0
            % task A
            f(i, n) = linear_f_offsets(n) + uSf(i, 1) * linear_f_slopes(1, n);
        else
            % task B
            f(i, n) = linear_f_offsets(n) + uSf(i, 2) * linear_f_slopes(2, n);
        end
    end
end

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
    'S', S, 'D', D, 'Sf', Sf, 'tauf', 1, 'signs', logspace(-1,.3,N), 'stim_dist_fun', @twoTaskDistFun);

disp('generate');
[simData, xTrue, fTrue] = gpfa.simulate([], f);
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
iZero = find(all(gpfa.uSf == 0, 2));
iTaskA = unique([find(gpfa.uSf(:,1) ~= 0); iZero]);
iTaskB = unique([find(gpfa.uSf(:,2) ~= 0); iZero]);
for n=1:N
    plot(gpfa.uSf(iTaskA, 1), fTrue(iTaskA, n), 'LineWidth', 2, 'Color', colors(n, :));
    plot(gpfa.uSf(iTaskB, 2), fTrue(iTaskB, n), 'LineWidth', 2, 'Color', colors(n, :));
    errorbar(gpfa.uSf(iTaskA, 1), mu_f(iTaskA, n), sqrt(diag(sigma_f{n}(iTaskA, iTaskA))), 'Color', colors(n, :));
    errorbar(gpfa.uSf(iTaskB, 2), mu_f(iTaskB, n), sqrt(diag(sigma_f{n}(iTaskB, iTaskB))), 'Color', colors(n, :));
end

%% Test inference at 'queried' points

disp('query');
queryTimes = gpfa.times + 0.5;
queryF = stims;
queryZero = all(queryF == 0, 2);
queryA = queryF(:,1) ~= 0 | queryZero;
queryB = queryF(:,2) ~= 0 | queryZero;
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
    errorbar(gpfa.uSf(iTaskA, 1), mu_f(iTaskA, n), sqrt(diag(sigma_f{n}(iTaskA, iTaskA))), '-o', 'Color', colors(n, :));
    errorbar(gpfa.uSf(iTaskB, 2), mu_f(iTaskB, n), sqrt(diag(sigma_f{n}(iTaskB, iTaskB))), '-o', 'Color', colors(n, :));
    errorbar(queryF(queryA,1), mu_f_q(queryA, n), sqrt(diag(sigma_f_q{n}(queryA,queryA))), 'Color', colors(n, :)/2);
    errorbar(queryF(queryB,2), mu_f_q(queryB, n), sqrt(diag(sigma_f_q{n}(queryB,queryB))), 'Color', colors(n, :)/2);
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

function dist = twoTaskDistFun(s1, s2)
% Distance logic is as follows: (i) if s1 and s2 are both [0 0] then distance is 0, (ii) if either
% s1 or s2 is zero, use standard distance, (iii) if s1 and s2 are nonzero in the same index, we just
% use abs difference, (iv) otherwise distance is considered infinite 'across tasks'
if all([s1 s2] == 0)
    dist = 0;
elseif all(s1 == 0) || all(s2 == 0)
    dist = sum(abs(s1 - s2));
elseif all((s1.*s2) == 0)
    dist = inf;
else
    dist = sum(abs(s1 - s2));
end
end
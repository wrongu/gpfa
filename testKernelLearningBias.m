%% Generate ground truth data

N = 30;
T = 100;
L = [1 2 3];
nL = length(L);
runs = 20;

true_taus = zeros(runs, nL, max(L));
fit_taus = zeros(runs, nL, max(L));
true_rhos = zeros(runs, nL, max(L));
fit_rhos = zeros(runs, nL, max(L));

for iL=1:nL
    l = L(iL);
    for iRun=1:runs
        saveFile = fullfile('test_runs', sprintf('kernelBias_L%d_run%03d_fixrho.mat', l, iRun));
        if exist(saveFile, 'file')
            load(saveFile);
        else
            [tt, ft, tr, fr] = doRun(N, l, T);
            save(saveFile, 'tt', 'ft', 'tr', 'fr');
        end
        
        padding = nan(1, size(true_taus, 3) - l);
        true_taus(iRun, iL, :) = [tt padding];
        fit_taus(iRun, iL, :) = [ft padding];
        true_rhos(iRun, iL, :) = [tr padding];
        fit_rhos(iRun, iL, :) = [fr padding];
    end
end

%% Plot

if ~exist('subplotsquare', 'file'), addpath('~/Research/tools'); end

for iL=1:nL
    subplot(2, nL, iL);
    scatter(flatten(true_taus(:, iL, :)), flatten(fit_taus(:, iL, :)));
    title(sprintf('L = %d', L(iL)));
    xlabel('true \tau');
    ylabel('fit \tau');
    axis equal;
    hold on;
    plot(xlim, xlim);
    
    subplot(2, nL, iL+nL);
    scatter(flatten(true_rhos(:, iL, :)), flatten(fit_rhos(:, iL, :)));
    title(sprintf('L = %d', L(iL)));
    xlabel('true \rho');
    ylabel('fit \rho');
    axis equal;
    hold on;
    plot(xlim, xlim);
end

%% Helper function

function v = flatten(v)
v = v(:);
end

function [true_taus, fit_taus, true_rhos, fit_rhos] = doRun(N, L, T)
R = .3 * exp(randn(N, 1) * .02);
sigs = ones(1, L);
true_taus = rand(1, L) * 45 + 5;
true_rhos = exprnd(.01, 1, L);

b = 10 * rand(N, 1);
C = zeros(N, L);
for l=1:L
    C(:, l) = sin(linspace(0, 2 * pi, N) + l);
end

disp('init');
fakeData = zeros(T, N);
gpfa = GPFA(fakeData, L, 'dt', 1, 'R', R, 'taus', true_taus, 'sigs', sigs, 'rhos', true_rhos, 'C', C);

disp('generate');
simData = gpfa.simulate();
gpfa = gpfa.setFields('Y', simData);
init = gpfa.setFields('fixed', {'C', 'R', 'b', 'rhos'}, 'lr', 5e-4, 'rhos', true_rhos);
[bestFit, ~] = init.fitEM(500, 1e-6);

fit_taus = bestFit.taus;
fit_rhos = bestFit.rhos;
end
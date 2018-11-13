function [gpfaObj, Q] = emStep(gpfaObj, itr)
R = gpfaObj.R;
T = gpfaObj.T;
L = gpfaObj.L;
C = gpfaObj.C;
D = gpfaObj.D;
Y = gpfaObj.Y;
b = gpfaObj.b;
Gamma = gpfaObj.Gamma;

if ~isempty(gpfaObj.S)
    stim_predict = gpfaObj.S * D';
else
    stim_predict = 0;
end

%% E-Step

[mu_x, sigma_x, e_xx] = gpfaObj.inferX();

% sigma_tt will contain values of sigma_x at all t1==t2
sigma_tt = zeros(L, L, T);
for t=1:T
    sigma_tt(:, :, t) = sigma_x(t:T:T*L, t:T:T*L);
end
% Compute E[x'x] from sigma_tt
expected_xx = (mu_x' * mu_x) + sum(sigma_tt, 3);

y_resid = Y - b' - stim_predict;
y_resid(isnan(y_resid)) = 0;
y_resid_Ri = y_resid ./ R';

logdet_R = sum(log(R));
trace_gamma_sigma = sum(sum(Gamma' .* sigma_x));
neg_2Q = T * logdet_R + sum(sum(y_resid .* y_resid_Ri)) - 2 * sum(sum(y_resid_Ri .* (mu_x * C'))) ...
    + vec(mu_x)' * Gamma * vec(mu_x) + trace_gamma_sigma;
Q = -1/2 * neg_2Q;

%% M-Step

% Note that the 'true' M-Step would jointly optimize b, C, D, and R together. We approximate this
% here by updating in the order b, C, D, R, since R depends on the previous 3, and all depend on b.

if ~any(strcmp('b', gpfaObj.fixed))
    residual = Y - mu_x * C' - stim_predict;
    b = nanmean(residual, 1)';
end

if ~any(strcmp('C', gpfaObj.fixed))
    residual = Y - b' - stim_predict;
    residual(isnan(residual)) = 0;
    C = (residual' * mu_x) / expected_xx;
end

if ~any(strcmp('D', gpfaObj.fixed)) && ~isempty(gpfaObj.S)
    residual = Y - b' - mu_x * C';
    residual(isnan(residual)) = 0;
    D = residual' * gpfaObj.S / (gpfaObj.S' * gpfaObj.S);
end

if ~any(strcmp('R', gpfaObj.fixed))
    residual = (Y - b' - mu_x * C' - stim_predict);
    residual(isnan(residual)) = 0;
    R = diag((residual' * residual + C * sum(sigma_tt, 3) * C') / T);
end

prev_rhos = gpfaObj.rhos;
prev_taus = gpfaObj.taus;
if ~any(strcmp('rhos', gpfaObj.fixed)) || ~any(strcmp('taus', gpfaObj.fixed))
    [QK, ~, ~] = gpfaObj.timescaleDeriv(e_xx);
    Q = Q + QK;
    lr = gpfaObj.lr * (1/2)^((itr-1) / gpfaObj.lr_decay);
    % Perform some number of gradient steps on timescales
    for step=1:25
        % Get gradient
        [~, dQ_dlogtau2, dQ_dlogrho2] = gpfaObj.timescaleDeriv(e_xx);
        
        % Step tau
        gpfaObj.log_tau2s = gpfaObj.log_tau2s + lr * dQ_dlogtau2;
        gpfaObj.taus = exp(gpfaObj.log_tau2s / 2);
        
        % Step rho
        gpfaObj.log_rho2s = gpfaObj.log_rho2s + lr * dQ_dlogrho2;
        gpfaObj.rhos = exp(gpfaObj.log_rho2s / 2);
        
        % Update GP covariance matrix
        gpfaObj = gpfaObj.updateK();
    end
end

if any(strcmp('taus', gpfaObj.fixed))
    gpfaObj.taus = prev_taus;
    gpfaObj.log_tau2s = 2 * log(gpfaObj.taus);
end

if any(strcmp('rhos', gpfaObj.fixed))
    gpfaObj.rhos = prev_rhos;
    gpfaObj.log_rho2s = 2 * log(gpfaObj.rhos);
end

gpfaObj.b = b;
gpfaObj.C = C;
gpfaObj.D = D;
gpfaObj.R = R;

%% Update precomputed matrices
gpfaObj = gpfaObj.updateAll();

end

function x = vec(x)
x = x(:);
end
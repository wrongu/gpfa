function [gpfaObj, Q, H] = emStep(gpfaObj, itr)
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

if isempty(gpfaObj.Sf)
    [mu_x, sigma_x] = gpfaObj.inferX();
else
    [mu_x, sigma_x, mu_f, sigma_f] = gpfaObj.inferMeanFieldXF();
end

if ~isempty(gpfaObj.Sf)
    stim_predict = stim_predict + mu_f(gpfaObj.Sf_ord, :);
end

% sigma_tt will contain values of sigma_x at all t1==t2
sigma_tt = zeros(L, L, T);
for t=1:T
    sigma_tt(:, :, t) = sigma_x(t:T:T*L, t:T:T*L);
end
% Compute E[x'x] from sigma_tt
e_xx_inner = (mu_x' * mu_x) + sum(sigma_tt, 3);

y_resid = Y - b' - stim_predict;
y_resid(isnan(y_resid)) = 0;
y_resid_Ri = y_resid ./ R';

logdet_R = sum(log(2*pi*R));
trace_gamma_sigma = sum(sum(Gamma' .* sigma_x));
Q = -1/2 * (T * logdet_R + sum(sum(y_resid .* y_resid_Ri)) - 2 * sum(sum(y_resid_Ri .* (mu_x * C'))) ...
    + vec(mu_x)' * Gamma * vec(mu_x) + trace_gamma_sigma);

H = 1/2*logdet(2*pi*exp(1)*sigma_x);

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
    C = (residual' * mu_x) / e_xx_inner;
end

if ~any(strcmp('D', gpfaObj.fixed)) && ~isempty(gpfaObj.S)
    residual = Y - b' - mu_x * C';
    if ~isempty(gpfaObj.Sf)
        residual = residual - mu_f(gpfaObj.Sf_ord, :);
    end
    residual(isnan(residual)) = 0;
    D = residual' * gpfaObj.S / (gpfaObj.S' * gpfaObj.S);
end

if ~any(strcmp('R', gpfaObj.fixed))
    residual = (Y - b' - mu_x * C' - stim_predict);
    residual(isnan(residual)) = 0;
    R = diag((residual' * residual + C * sum(sigma_tt, 3) * C') / T);
end

update_tau = ~any(strcmp('taus', gpfaObj.fixed));
update_rho = ~any(strcmp('rhos', gpfaObj.fixed));

if update_tau || update_rho
    [QK, ~, ~] = gpfaObj.timescaleDeriv(mu_x, sigma_x);
    Q = Q + QK;
    lr = gpfaObj.lr * (1/2)^((itr-1) / gpfaObj.lr_decay);
    
    % Perform some number of gradient steps on timescales
    for step=1:25
        % Get gradient
        [~, dQ_dlogtau2, dQ_dlogrho2] = gpfaObj.timescaleDeriv(mu_x, sigma_x);
        
        % Step tau
        if ~any(strcmp('taus', gpfaObj.fixed))
            gpfaObj.log_tau2s = gpfaObj.log_tau2s + lr * dQ_dlogtau2;
            gpfaObj.taus = exp(gpfaObj.log_tau2s / 2);
        end
        
        % Step rho
        if ~any(strcmp('rhos', gpfaObj.fixed))
            gpfaObj.log_rho2s = gpfaObj.log_rho2s + lr * dQ_dlogrho2;
            gpfaObj.rhos = exp(gpfaObj.log_rho2s / 2);
        end
        
        % Update K for next iteration (note: important that we only update K and not Cov here, as
        % any update to expectations of x changes the underlying Q we're optimizing).
        gpfaObj = gpfaObj.updateK();
        
        % Break when changes get small
        if abs(lr*dQ_dlogtau2) + abs(lr*dQ_dlogrho2) < 1e-9
            break
        end
    end
end

if ~isempty(gpfaObj.Sf)
    Qf = 0;
    Hf = 0;
    S = length(gpfaObj.Ns);
    logdetK = logdet(gpfaObj.Kf);
    for n=gpfaObj.N:-1:1
        e_ff_n{n} = sigma_f{n} + mu_f(:,n)*mu_f(:,n)';
        Qf = Qf - 1/2*(trace((gpfaObj.signs(n)^2 * gpfaObj.Kf) \ e_ff_n{n}) + gpfaObj.signs(n)^(2*S)*logdetK);
        Hf = Hf + 1/2*logdet(2*pi*exp(1)*sigma_f{n});
    end
    Q = Q + Qf;
    H = H + Hf;
end

if ~isempty(gpfaObj.Sf) && ~any(strcmp('signs', gpfaObj.fixed))
    warning('skipping signs update since it''s broken right now');
    % for n=1:gpfaObj.N
    %     frac = trace(gpfaObj.Kf \ e_ff_n{n}) / (S * logdetK);
    %     gpfaObj.signs(n) = frac^(-2*S-2);
    % end
end

if ~isempty(gpfaObj.Sf) && ~any(strcmp('tauf', gpfaObj.fixed))
    lr = gpfaObj.lr * (1/2)^((itr-1) / gpfaObj.lr_decay);
    
    logtauf2 = 2*log(gpfaObj.tauf);
    
    % Perform some number of gradient steps on tau_f
    for step=1:25
        % Get gradient
        dQ_dlogtauf2 = gpfaObj.stimScaleDeriv(mu_f, sigma_f);
        
        % Step tau_f
        logtauf2 = logtauf2 + lr * dQ_dlogtauf2;
        gpfaObj.tauf = exp(logtauf2 / 2);
        
        % Update Kf for next iteration
        gpfaObj = gpfaObj.updateKernelF();
        
        % Break when changes get small
        if abs(lr*dQ_dlogtauf2) < 1e-9
            break
        end
    end
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
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

% Get the sum of the variances (covariance diagonals) of each latent
variances = cellfun(@(sig) sum(diag(sig)), sigma_x);
% Compute E[x'x] under the factorized posterior (zero covariance between latents by assumption)
e_xx_inner = (mu_x' * mu_x) + diag(variances);

y_resid = Y - b' - stim_predict;
y_resid(isnan(y_resid)) = 0;
y_resid_Ri = y_resid ./ R';

logdet_R = sum(log(2*pi*R));
trace_gamma_sigma = sum(arrayfun(@(l) Gamma{l,l}(:)'*sigma_x{l}(:), 1:L));
Q = -1/2 * (T * logdet_R + sum(sum(y_resid .* y_resid_Ri)) - 2 * sum(sum(y_resid_Ri .* (mu_x * C'))) ...
    + vec(mu_x)' * cell2mat(Gamma) * vec(mu_x) + trace_gamma_sigma);

H = sum(arrayfun(@(l) 1/2*logdet(2*pi*exp(1)*sigma_x{l}), 1:L));

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
    cov_y_x = C * diag(variances) * C' / T;
    if isempty(gpfaObj.Sf)
        cov_y_f = 0;
    else
        var_y_f = cellfun(@(sig_f) dot(gpfaObj.Ns, diag(sig_f)), sigma_f);
        cov_y_f = diag(var_y_f) / T;
    end
    R = diag((residual' * residual) / T + cov_y_x + cov_y_f);
end

update_tau = ~any(strcmp('taus', gpfaObj.fixed));
update_rho = ~any(strcmp('rhos', gpfaObj.fixed));

if (update_tau || update_rho) && mod(itr, gpfaObj.kernel_update_freq) == 0
    [QK, ~, ~] = gpfaObj.timescaleDeriv(mu_x, sigma_x);
    Q = Q + QK;
    lr = gpfaObj.lr * (1/2)^((itr-1) / gpfaObj.lr_decay);

    logtau2s = 2*log(gpfaObj.taus);
    logrho2s = 2*log(gpfaObj.rhos);
    
    % Perform some number of gradient steps on timescales
    for step=1:25
        % Get gradient
        [~, dQ_dlogtau2, dQ_dlogrho2] = gpfaObj.timescaleDeriv(mu_x, sigma_x);
        
        % Step tau
        if ~any(strcmp('taus', gpfaObj.fixed))
            logtau2s = logtau2s + lr * dQ_dlogtau2;
            gpfaObj.taus = exp(logtau2s / 2);
        end
        
        % Step rho
        if ~any(strcmp('rhos', gpfaObj.fixed))
            logrho2s = logrho2s + lr * dQ_dlogrho2;
            gpfaObj.rhos = exp(logrho2s / 2);
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
    e_ff_n = cell(1, gpfaObj.N);
    for n=gpfaObj.N:-1:1
        e_ff_n{n} = sigma_f{n} + mu_f(:,n)*mu_f(:,n)';
        Qf = Qf - 1/2*(trace((gpfaObj.signs(n)^2 * gpfaObj.Kf) \ e_ff_n{n}) + gpfaObj.signs(n)^(2*S)*logdetK);
        Hfn = 1/2*logdet(2*pi*exp(1)*sigma_f{n});
        % Numerical precision and lack of regularization on sigma_f means near-zero determinant
        % cases appear like imaginary Hfn and negative determinant. Clip Hfn here.
        Hf = Hf + max(0, real(Hfn));
    end
    Q = Q + Qf;
    H = H + Hf;
end

if ~isempty(gpfaObj.Sf) && ~any(strcmp('signs', gpfaObj.fixed)) && mod(itr, gpfaObj.kernel_update_freq) == 0
    warning('Learning of ''signs'' not stable yet. Skipping.');
    % dimf = length(gpfaObj.Ns);
    % for n=1:gpfaObj.N
    %     gamma_n = 1/(dimf+1)*(log(trace(gpfaObj.Kf \ e_ff_n{n})) - log(dimf) - logdetK);
    %     gpfaObj.signs(n) = exp(gamma_n / 2);
    % end
end

    function [nQf, gradNegQf] = negQf(logtauf2)
        tmpObj = gpfaObj;
        tmpObj.tauf = exp(logtauf2/2);
        tmpObj = tmpObj.updateKernelF();
        newLogDetK = logdet(tmpObj.Kf);
        if ~isreal(newLogDetK), newLogDetK = 0; end
        nQf = sum(arrayfun(@(n) 1/2*(trace((tmpObj.signs(n)^2 * tmpObj.Kf) \ e_ff_n{n}) + tmpObj.signs(n)^(2*S)*newLogDetK), 1:tmpObj.N));
        gradNegQf = -tmpObj.stimScaleDeriv(mu_f, sigma_f);
    end

if ~isempty(gpfaObj.Sf) && ~any(strcmp('tauf', gpfaObj.fixed)) && mod(itr, gpfaObj.kernel_update_freq) == 0
    % In practice, gradient steps with learning rate 'lr' was found to be unstable for all parameter
    % regimes tested. fminunc is a bit slower but has better guarantees.
    opts = optimoptions('fminunc', 'Algorithm', 'trust-region', 'SpecifyObjectiveGradient', true, ...
        'Display', 'none');
    [newLogTauf, newNegQf] = fminunc(@negQf, 2*log(gpfaObj.tauf), opts);
    assert(-newNegQf >= Qf, 'tauf optimization failed!');
    gpfaObj.tauf = exp(newLogTauf / 2);
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
function [gpfaObj, Q, H] = emStep(gpfaObj, itr)
if ~gpfaObj.initialized, gpfaObj = gpfaObj.updateAll(); end

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

missing_data = isnan(gpfaObj.Y);

%% E-Step

if isempty(gpfaObj.Sf)
    [mu_x, sigma_x] = gpfaObj.inferX();
else
    % Convergence tolerance for mu_x and mu_f gets more strict as iterations go on
    tolerance = eps + 1e-3 * (1/2)^((itr-1) / gpfaObj.lr_decay);
    [mu_x, sigma_x, mu_f, sigma_f] = gpfaObj.inferMeanFieldXF([], [], [], tolerance);
end

if ~isempty(gpfaObj.Sf)
    for k=1:gpfaObj.nGP
        stim_predict = stim_predict + mu_f{k}(gpfaObj.Sf_ord{k}, :);
    end
end

% Get the sum of the variances (covariance diagonals) of each latent
variances = cellfun(@(sig) sum(diag(sig)), sigma_x);
% Compute E[x'x] under the factorized posterior (zero covariance between latents by assumption)
e_xx_inner = (mu_x' * mu_x) + diag(variances);

y_resid = Y - b' - stim_predict;
y_resid(isnan(y_resid)) = 0;
y_resid_Ri = y_resid ./ R';

logdet_R = sum(log(2*pi*R));
if gpfaObj.L > 0
    xGx = 0;
    trace_gamma_sigma = 0;
    for l1=1:gpfaObj.L
        % Note: Gamma is a diagonal matrix, so trace(G*S) is dot(diag(G),diag(S))
        trace_gamma_sigma = trace_gamma_sigma + dot(diag(Gamma{l1,l1}), diag(sigma_x{l1}));
        for l2=1:gpfaObj.L
            xGx = xGx + mu_x(:,l1)' * Gamma{l1, l2} * mu_x(:,l2);
        end
    end

    Q = -1/2 * (T * logdet_R + sum(sum(y_resid .* y_resid_Ri)) - 2 * sum(sum(y_resid_Ri .* (mu_x * C'))) ...
        + xGx + trace_gamma_sigma);
else
    Q = -1/2 * (T * logdet_R + sum(sum(y_resid .* y_resid_Ri)));
end

% Add kernel component to Q
Q = Q + gpfaObj.timescaleQ(mu_x, sigma_x);

H = sum(arrayfun(@(l) 1/2*logdet(2*pi*exp(1)*sigma_x{l}), 1:L));

%% M-Step

% Note that the 'true' M-Step would jointly optimize b, C, D, and R together. We approximate this
% here by updating in the order b, C, D, R, since R depends on the previous 3, and all depend on b.

if gpfaObj.L > 0
    xC = mu_x * C';
else
    xC = 0;
end

if ~any(strcmp('b', gpfaObj.fixed))
    residual = Y - xC - stim_predict;
    b = nanmean(residual, 1)';
end

if ~any(strcmp('C', gpfaObj.fixed)) && gpfaObj.L > 0
    residual = Y - b' - stim_predict;
    residual(missing_data) = 0;
    C = (residual' * mu_x) / e_xx_inner;
    xC = mu_x * C';
end

if ~any(strcmp('D', gpfaObj.fixed)) && ~isempty(gpfaObj.S)
    stim_predict = stim_predict - gpfaObj.S * D';
    residual = Y - b' - xC;
    if ~isempty(gpfaObj.Sf)
        for k=1:gpfaObj.nGP
            residual = residual - mu_f{k}(gpfaObj.Sf_ord{k}, :);
        end
    end
    residual(missing_data) = 0;
    D = residual' * gpfaObj.S / (gpfaObj.S' * gpfaObj.S);
    stim_predict = stim_predict + gpfaObj.S * D';
end

if ~any(strcmp('R', gpfaObj.fixed))
    residual = (Y - b' - xC - stim_predict);
    residual(missing_data) = 0;
    if gpfaObj.L > 0
        cov_y_x = C * diag(variances) * C' / T;
    else
        cov_y_x = 0;
    end
    if isempty(gpfaObj.Sf)
        cov_y_f = 0;
    else
        cov_y_f = zeros(gpfaObj.N);
        T_per_unit = sum(~missing_data, 1);
        for k=1:gpfaObj.nGP
            for n=1:gpfaObj.N
                var_y_f_n = gpfaObj.Ns{k}(n, :) * diag(sigma_f{k}{n});
                cov_y_f(n, n) = cov_y_f(n, n) + var_y_f_n ./ T_per_unit(n);
            end
        end
    end
    R = diag((residual' * residual) / T + cov_y_x + cov_y_f);
    R = max(R, 1e-6);
end

update_tau = ~any(strcmp('taus', gpfaObj.fixed));
update_rho = ~any(strcmp('rhos', gpfaObj.fixed));

    function [nQt, gradNegQt] = negQt(logtau2_logrho2)
        inner_logtau2s = logtau2_logrho2(1, :);
        inner_logrho2s = logtau2_logrho2(2, :);
        
        tmpObj = gpfaObj;
        
        if ~any(strcmp('taus', gpfaObj.fixed))
            tmpObj.taus = exp(inner_logtau2s / 2);
        end
        
        if ~any(strcmp('rhos', gpfaObj.fixed))
            tmpObj.rhos = exp(inner_logrho2s / 2);
        end
        
        tmpObj = tmpObj.updateK();
        
        nQt = double(-tmpObj.timescaleQ(mu_x, sigma_x));
        [dQ_dlogtau2, dQ_dlogrho2] = tmpObj.timescaleDeriv(mu_x, sigma_x);
        
        gradNegQt = double(vertcat(-dQ_dlogtau2 * update_tau, -dQ_dlogrho2 * update_rho));
    end

if (update_tau || update_rho) && mod(itr, gpfaObj.kernel_update_freq) == 1
    opts = optimoptions('fminunc', 'Algorithm', 'trust-region', 'SpecifyObjectiveGradient', true, ...
        'Display', 'none');
    logtau2s = 2*log(gpfaObj.taus);
    logrho2s = 2*log(gpfaObj.rhos);
    [newLogTau2LogRho2, ~] = fminunc(@negQt, vertcat(logtau2s, logrho2s), opts);
    
    gpfaObj.taus = exp(newLogTau2LogRho2(1, :) / 2);
    gpfaObj.rhos = exp(newLogTau2LogRho2(2, :) / 2);
end

if ~isempty(gpfaObj.Sf)
    Qf = 0;
    Hf = 0;
    for k=gpfaObj.nGP:-1:1
        S = size(gpfaObj.Ns{k}, 2);
        logdetK = logdet(gpfaObj.Kf{k});
        e_ff_kn = cell(gpfaObj.nGP, gpfaObj.N);
        for n=gpfaObj.N:-1:1
            e_ff_kn{k,n} = sigma_f{k}{n} + mu_f{k}(:,n)*mu_f{k}(:,n)';
            Qf = Qf - 1/2*(trace((gpfaObj.signs(k,n)^2 * gpfaObj.Kf{k}) \ e_ff_kn{k,n}) + gpfaObj.signs(k,n)^(2*S)*logdetK);
            Hfn = 1/2*logdet(2*pi*exp(1)*sigma_f{k}{n});
            % Numerical precision and lack of regularization on sigma_f means near-zero determinant
            % cases appear like imaginary Hfn and negative determinant. Clip Hfn here.
            Hf = Hf + max(0, real(Hfn));
        end
    end
    Q = Q + Qf;
    H = H + Hf;
end

if ~isempty(gpfaObj.Sf) && ~any(strcmp('signs', gpfaObj.fixed)) && mod(itr, gpfaObj.kernel_update_freq) == 1
    warning('Learning of ''signs'' not stable yet. Skipping.');
    % dimf = size(gpfaObj.Ns, 2);
    % for n=1:gpfaObj.N
    %     gamma_n = 1/(dimf+1)*(log(trace(gpfaObj.Kf \ e_ff_n{n})) - log(dimf) - logdetK);
    %     gpfaObj.signs(n) = exp(gamma_n / 2);
    % end
end

    function [nQf, gradNegQf] = negQf(logtauf2)
        tmpObj = gpfaObj;
        tmpObj.tauf = exp(logtauf2/2);
        tmpObj = tmpObj.updateKernelF();
        for kk=1:gpfaObj.nGP
            newLogDetK = logdet(tmpObj.Kf{kk});
            if ~isreal(newLogDetK), newLogDetK = 0; end
            nQf = sum(arrayfun(@(n) 1/2*(trace((tmpObj.signs(kk,n)^2 * tmpObj.Kf{kk}) \ e_ff_kn{kk,n}) + tmpObj.signs(kk,n)^(2*S)*newLogDetK), 1:tmpObj.N));
        end
        gradNegQf = -tmpObj.stimScaleDeriv(mu_f, sigma_f);
    end

if ~isempty(gpfaObj.Sf) && ~any(strcmp('tauf', gpfaObj.fixed)) && mod(itr, gpfaObj.kernel_update_freq) == 1
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

%% Ensure outputs are in CPU memory
Q = gather(Q);
H = gather(H);

end

function x = vec(x)
x = x(:);
end
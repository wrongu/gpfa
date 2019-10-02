function [gpfaObj, Q, H] = emStep(gpfaObj, itr, infer_tol)
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
    [mu_x, sigma_x, mu_f, sigma_f] = gpfaObj.inferMeanFieldXF([], [], [], infer_tol);
end

if ~isempty(gpfaObj.Sf)
    for k=1:gpfaObj.nGP
        stim_predict = stim_predict + mu_f{k}(gpfaObj.Sf_ord{k}, :);
    end
end

% Get the sum of the variances (covariance diagonals) of each latent, separate per neuron
% considering only where each neuron has data
for n=gpfaObj.N:-1:1
    mask = ~missing_data(:,n);
    variances{n} = cellfun(@(sig) mask'*diag(sig), sigma_x);
end

y_resid = Y - b' - stim_predict;
y_resid(missing_data) = 0;
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
    % Here, missing data appear as NaN values. Average over all non-missing values to estimate b.
    b = nanmean(residual, 1)';
    b = gpfaObj.getNewValueHandleConstraints('b', b);
end

if ~any(strcmp('C', gpfaObj.fixed)) && gpfaObj.L > 0
    residual = Y - b' - stim_predict;
    if ~any(missing_data(:))
        % Compute E[x'x] under the factorized posterior (zero covariance between latents by assumption)
        e_xx_inner = (mu_x' * mu_x) + diag(variances{1});
        C = (residual' * mu_x) / e_xx_inner;
    else
        % Which 'x' matter depends on the neuron: for loadings C(:,n), only consider where neuron n
        % has data.
        for n=1:gpfaObj.N
            mask = ~missing_data(:, n);
            C(n, :) = (residual(mask, n)' * mu_x(mask, :)) / (mu_x(mask,:)'*mu_x(mask,:) + variances{n});
        end
    end
    C = gpfaObj.getNewValueHandleConstraints('C', C);
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
    if ~any(missing_data(:))
        D = residual' * gpfaObj.S / (gpfaObj.S' * gpfaObj.S);
    else
        % Analogous to 'C' update, only consider trials of 'S' where neuron 'n' has data
        for n=1:gpfaObj.N
            mask = ~missing_data(:, n);
            S_inner = gpfaObj.S(mask, :)' * gpfaObj.S(mask, :);
            D(n, :) = residual(mask, n)' * gpfaObj.S(mask, :) / S_inner;
        end
    end
    D = gpfaObj.getNewValueHandleConstraints('D', D);
    stim_predict = stim_predict + gpfaObj.S * D';
end

if ~any(strcmp('R', gpfaObj.fixed))
    residual = (Y - b' - xC - stim_predict);
    residual(missing_data) = 0;
    T_per_unit = sum(~missing_data, 1);
    for n=1:gpfaObj.N
        % Get residual variance term due to uncertainty in latents
        if gpfaObj.L > 0
            cov_y_x = C * diag(variances{n}) * C' / T_per_unit(n);
        else
            cov_y_x = 0;
        end
        % Get residual variance term due to uncertainty in GP tuning
        if isempty(gpfaObj.Sf)
            cov_y_f = 0;
        else
            cov_y_f = 0;
            for k=1:gpfaObj.nGP
                % Note: Ns{k}(n,:) contains, for stimulus {k}, the number of times each stimulus
                % value appeared on trials where neuron n has data
                assert(sum(gpfaObj.Ns{k}(n, :)) == T_per_unit(n));
                var_y_f_n = gpfaObj.Ns{k}(n, :) * diag(sigma_f{k}{n});
                cov_y_f = cov_y_f + var_y_f_n ./ T_per_unit(n);
            end
        end
        % Combine variances
        R(n) = residual(:,n)' * residual(:,n) / T_per_unit(n) + cov_y_x + cov_y_f;
    end
    R = max(R, 1e-6);
    R = gpfaObj.getNewValueHandleConstraints('R', R);
end

update_tau = ~any(strcmp('taus', gpfaObj.fixed));
update_rho = ~any(strcmp('rhos', gpfaObj.fixed));
[~, tau_update_mask] = gpfaObj.getNewValueHandleConstraints('taus', []);
[~, rho_update_mask] = gpfaObj.getNewValueHandleConstraints('rhos', []);

    function [nQt, gradNegQt] = negQt(logtau2_logrho2)
        inner_logtau2s = logtau2_logrho2(1, :);
        inner_logrho2s = logtau2_logrho2(2, :);
        
        tmpObj = gpfaObj;
        
        % Update values of 'rhos' and 'taus' but only where they aren't constrained
        tmpObj.taus(tau_update_mask) = exp(inner_logtau2s(tau_update_mask) / 2);
        tmpObj.rhos(rho_update_mask) = exp(inner_logrho2s(rho_update_mask) / 2);
        
        tmpObj = tmpObj.updateK();
        
        nQt = double(-tmpObj.timescaleQ(mu_x, sigma_x));
        [dQ_dlogtau2, dQ_dlogrho2] = tmpObj.timescaleDeriv(mu_x, sigma_x);
        
        % Concatenate gradients, setting gradient to zero if a constraint is active
        gradNegQt = double(vertcat(-dQ_dlogtau2 .* tau_update_mask, -dQ_dlogrho2 .* rho_update_mask));
    end

if (update_tau || update_rho) && mod(itr, gpfaObj.kernel_update_freq) == 0
    opts = optimoptions('fminunc', 'Algorithm', 'trust-region', 'SpecifyObjectiveGradient', true, ...
        'Display', 'none');
    logtau2s = 2*log(gpfaObj.taus);
    logrho2s = 2*log(gpfaObj.rhos);
    [newLogTau2LogRho2, ~] = fminunc(@negQt, vertcat(logtau2s, logrho2s), opts);
    
    gpfaObj.taus(tau_update_mask) = exp(newLogTau2LogRho2(1, tau_update_mask) / 2);
    gpfaObj.rhos(rho_update_mask) = exp(newLogTau2LogRho2(2, rho_update_mask) / 2);
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

if ~isempty(gpfaObj.Sf) && ~any(strcmp('signs', gpfaObj.fixed)) && mod(itr, gpfaObj.kernel_update_freq) == 0
    warning('Learning of ''signs'' not stable yet. Skipping.');
    % dimf = size(gpfaObj.Ns, 2);
    % for n=gpfaObj.N:-1:1
    %     gamma_n(n) = 1/(dimf+1)*(log(trace(gpfaObj.Kf \ e_ff_n{n})) - log(dimf) - logdetK);
    % end
    % gpfaObj.signs = gpfaObj.getNewValueHandleConstraints('signs', exp(gamma_n / 2));
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

if ~isempty(gpfaObj.Sf) && ~any(strcmp('tauf', gpfaObj.fixed)) && mod(itr, gpfaObj.kernel_update_freq) == 0
    % In practice, gradient steps with learning rate 'lr' was found to be unstable for all parameter
    % regimes tested. fminunc is a bit slower but has better guarantees.
    opts = optimoptions('fminunc', 'Algorithm', 'trust-region', 'SpecifyObjectiveGradient', true, ...
        'Display', 'none');
    [newLogTauf, newNegQf] = fminunc(@negQf, 2*log(gpfaObj.tauf), opts);
    assert(-newNegQf >= Qf, 'tauf optimization failed!');
    gpfaObj.tauf = gpfaObj.getNewValueHandleConstraints('tauf', exp(newLogTauf / 2));
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
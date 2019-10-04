function [Q, H, FVE, LL] = metrics(gpfaObj, latent_values)
%GPFA.METRICS compute goodness-of-fit metrics.
%
%Inputs: GPFA.metrics(latent_values) where latent_values is an optional struct containing mu_x,
%	mu_f, sig_x, and sig_f. If latent_values are not given, calls inferX or inferMeanFieldXF.
%
%Outputs: [Q, H, FVE, LL] defined as follows:
%	Q   = E_{p(latents|old params)}[log joint]
%   H   = entropy of latent
%   FVE = fraction of variance explained, in [0, 1] for all sane cases, but can go < 0 if
%       predictions somehow add variance..
%   LL  = not-quite log likelihood of model parameters, which is equal to Q+H+KL, where KL =
%   	divergence between inferred parameter values (mean field approximation) and true posterior.
%		We pretend here that KL=0, so trivially LL=Q+H

if ~gpfaObj.initialized, gpfaObj = gpfaObj.updateAll(); end

R = gpfaObj.R;
T = gpfaObj.T;
L = gpfaObj.L;
C = gpfaObj.C;
D = gpfaObj.D;
Y = gpfaObj.Y;
b = gpfaObj.b;
Gamma = gpfaObj.Gamma;

%% Compute Q and H

if ~isempty(gpfaObj.S)
    stim_predict = gpfaObj.S * D';
else
    stim_predict = 0;
end

missing_data = isnan(gpfaObj.Y);

if isempty(gpfaObj.Sf)
    if exist('latent_values', 'var')
        mu_x = latent_values.mu_x;
        sigma_x = latent_values.sigma_x;
    else
        [mu_x, sigma_x] = gpfaObj.inferX();
    end
else
    if exist('latent_values', 'var')
        mu_x = latent_values.mu_x;
        sigma_x = latent_values.sigma_x;
        mu_f = latent_values.mu_f;
        sigma_f = latent_values.sigma_f;
    else
        [mu_x, sigma_x, mu_f, sigma_f] = gpfaObj.inferMeanFieldXF();
    end
end

if ~isempty(gpfaObj.Sf)
    for k=1:gpfaObj.nGP
        stim_predict = stim_predict + mu_f{k}(gpfaObj.Sf_ord{k}, :);
    end
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

if ~isempty(gpfaObj.Sf)
    Qf = 0;
    Hf = 0;
    e_ff_kn = cell(gpfaObj.nGP, gpfaObj.N);
    for k=gpfaObj.nGP:-1:1
        S = size(gpfaObj.Ns{k}, 2);
        logdetK = logdet(gpfaObj.Kf{k});
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

% Ensure outputs are in CPU memory
Q = gather(Q);
H = gather(H);

%% Get LL

LL = Q + H;

%% Get FVE

if nargout >= 4
    varY = nanvar(gpfaObj.Y, [], 1);
    predY = gpfaObj.predictY(mu_x, mu_f);
    varResidY = nanvar(gpfaObj.Y - predY, [], 1);
    FVE = gather(1 - varResidY / varY);
end

end
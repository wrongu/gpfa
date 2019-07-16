function [mu_Y, mu_Ysq] = predictY(gpfaObj, mu_x, mu_f)

if ~gpfaObj.initialized, gpfaObj = gpfaObj.updateAll(); end

if isempty(gpfaObj.Sf)
    if ~exist('mu_x', 'var')
        mu_x = gpfaObj.inferX();
    end
else
    if ~exist('mu_x', 'var') || ~exist('mu_f', 'var')
        [mu_x, ~, mu_f, ~] = gpfaObj.inferMeanFieldXF();
    end
end

if gpfaObj.L > 0
    mu_Y = gpfaObj.b' + mu_x * gpfaObj.C';
else
    mu_Y = repmat(gpfaObj.b', gpfaObj.T, 1);
end

if ~isempty(gpfaObj.S)
    mu_Y = mu_Y + gpfaObj.S * gpfaObj.D';
end

if ~isempty(gpfaObj.Sf)
    for k=1:gpfaObj.nGP
        mu_Y = mu_Y + mu_f{k}(gpfaObj.Sf_ord{k}, :);
    end
end

if nargout >= 2
    % Using E[Y^2] = E[Y]^2 + var(Y), and var(Y) being the sum of variances due to each latent term
    [~, sig_x, ~, sig_f] = gpfaObj.inferMeanFieldXF();
    
    mu_Ysq = mu_Y.^2;
    
    for l=1:gpfaObj.L
        mu_Ysq = mu_Ysq + diag(sig_x{l}) * gpfaObj.C(:, l).^2;
    end
    
    for k=1:gpfaObj.nGP
        for n=1:gpfaObj.N
            var_f = diag(sig_f{k}{n});
            mu_Ysq(:,n) = mu_Ysq(:,n) + var_f(gpfaObj.Sf_ord{k});
        end
    end
end
end
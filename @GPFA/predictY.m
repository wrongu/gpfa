function [mu_Y] = predictY(gpfaObj, mu_x, mu_f)

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

end
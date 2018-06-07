function [mu_Y] = predictY(gpfaObj, mu_x)

if ~exist('mu_x', 'var'), mu_x = gpfaObj.inferX(); end
mu_Y = gpfaObj.b' + mu_x * gpfaObj.C';

if ~isempty(gpfaObj.S)
    mu_Y = mu_Y + gpfaObj.S * gpfaObj.D';
end

end
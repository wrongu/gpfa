function [mu_Y] = predictY(gpfaObj)

mu_x = gpfaObj.inferX();
mu_Y = gpfaObj.b' + mu_x * gpfaObj.C';

if ~isempty(gpfaObj.S)
    mu_Y = mu_Y + gpfaObj.S * gpfaObj.D';
end

end
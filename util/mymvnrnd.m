function [r,T] = mymvnrnd(mu,sigma,cases,T)
try
    if nargin == 3
        [r,T] = mvnrnd(mu, sigma, cases);
    else
        [r,T] = mvnrnd(mu,sigma,cases,T);
    end
catch
    % Thanks to https://stats.stackexchange.com/a/193735/234036
    [U,S,~] = svd(sigma);
    rS = sqrt(S);
    T = U*rS;
    r = mvnrnd(mu,sigma,cases,T);
end
end
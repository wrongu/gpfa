function [bestFit, Qs, Hs, converged] = fitEM(gpfaObj, maxIters, convergenceTol, startIter)

if ~exist('convergenceTol', 'var'), convergenceTol = 1e-6; end

if ~exist('startIter', 'var'), startIter = 1; end
numIters = (maxIters-startIter)+1;

Qs = zeros(1, numIters);
Hs = zeros(1, numIters);
tstart = tic;
converged = false;

allParams = {'R', 'C', 'D', 'b', 'taus', 'rhos', 'tauf', 'signs'};
lastParamValues = concatAllParams(gpfaObj, allParams);

for idx=1:numIters
    itr = startIter+idx-1;

    % TODO (here or inside emStep): block updates of parameters that depend on each other
    [newObj, Qs(idx), Hs(idx)] = gpfaObj.emStep(itr);
    elapsed = toc(tstart);
    
    newParamValues = concatAllParams(newObj, allParams);
    delta = norm(lastParamValues - newParamValues);
    if delta < convergenceTol
        converged = true;
        break
    end
    
    if idx >= 2
        deltaQ = (Qs(idx)+Hs(idx))-(Qs(idx-1)+Hs(idx-1));
    else
        deltaQ = inf;
    end
    
    fprintf('EM iteration %d/%d\tQ = %.2e\tdelta Q = %.2e\tdelta params = %.2e\ttime per iteration = %.2fs\n', ...
        itr, maxIters, Qs(idx)+Hs(idx), deltaQ, delta, elapsed / idx);
    
    gpfaObj = newObj;
    lastParamValues = newParamValues;
end

if ~converged
    warning('EM reached max iterations; potentially exiting before convergence');
else
    Qs = Qs(1:idx);
    Hs = Hs(1:idx);
end

bestFit = gpfaObj;
[~, Qs(end+1), Hs(end+1)] = bestFit.emStep(maxIters);

end

function v = concatAllParams(obj, params)
v = zeros(sum(cellfun(@(para) numel(obj.(para)), params)), 1);
j = 1;
for i=1:length(params)
    value = obj.(params{i});
    v(j:j+numel(value) - 1) = value(:);
    j = j + numel(value);
end
end
function [bestFit, Qs] = fitEM(gpfaObj, maxIters, convergenceTol, fixedParams)

if ~exist('convergenceTol', 'var'), convergenceTol = 1e-6; end
if ~exist('fixedParams', 'var'), fixedParams = {}; end

Qs = zeros(1, maxIters);
tstart = tic;

allParams = {'R', 'C', 'D', 'b'};
lastParamValues = concatAllParams(gpfaObj, allParams);

for itr=1:maxIters
    % TODO (here or inside emStep): block updates of parameters that depend on each other
    [newObj, Qs(itr)] = gpfaObj.emStep(fixedParams);
    elapsed = toc(tstart);
    
    fprintf('EM iteration %d/%d\tQ = %.2es\ttime per iteration = %.2fs\n', ...
        itr, maxIters, Qs(itr), elapsed / itr);
    
    newParamValues = concatAllParams(newObj, allParams);
    delta = norm(lastParamValues - newParamValues);
    if delta < convergenceTol
        break
    end
    
    gpfaObj = newObj;
    lastParamValues = newParamValues;
end

if itr == maxIters
    warning('EM reached max iterations; potentially exiting before convergence');
else
    Qs = Qs(1:itr);
end

bestFit = gpfaObj;
[~, Qs(end+1)] = bestFit.emStep();

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
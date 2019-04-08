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
lastDeltas = zeros(1, gpfaObj.kernel_update_freq);

for idx=1:numIters
    itr = startIter+idx-1;

    % TODO (here or inside emStep): block updates of parameters that depend on each other
    [newObj, Qs(idx), Hs(idx)] = gpfaObj.emStep(itr);
    elapsed = toc(tstart);
    
    newParamValues = concatAllParams(newObj, allParams);
    delta = norm(lastParamValues - newParamValues);
    lastDeltas = [lastDeltas(2:end) delta];
    if max(lastDeltas) < convergenceTol
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

    if deltaQ < 0
        converged = true;
        break
        % DEBUGGING
        % changedParams = setdiff(allParams, gpfaObj.fixed);
        % for iParam=1:length(changedParams)
        %     tmpObj = gpfaObj;
        %     tmpObj.fixed = setdiff(allParams, changedParams{iParam});
        %     [newTmpObj, oldQ, oldH] = tmpObj.emStep(itr);
        %     [~, newQ, newH] = newTmpObj.emStep(itr+1);
        %     deltaParam = norm(concatAllParams(newTmpObj, allParams) - concatAllParams(tmpObj, allParams));
        %     fprintf('%s change by magnitude %.2e, delta LL = %.2e\n', changedParams{iParam}, deltaParam, (newQ+newH)-(oldQ+oldH));
        % end
        % keyboard;
    end
    
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
v = zeros(sum(cellfun(@(para) mynumel(obj.(para)), params)), 1);
j = 1;
for i=1:length(params)
    value = obj.(params{i});
    if iscell(value)
        for k=1:length(value)
            v(j:j+numel(value{k}) - 1) = value{k}(:);
            j = j+numel(value{k});
        end
    else
        v(j:j+numel(value) - 1) = value(:);
        j = j + numel(value);
    end
end
end

function n = mynumel(v)
if iscell(v)
    n = sum(cellfun(@mynumel, v));
else
    n = numel(v);
end
end
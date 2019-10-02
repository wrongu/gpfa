function [bestFit, Qs, Hs, FVEs, converged] = fitEM(gpfaObj, maxIters, convergenceTol, startIter)

if ~gpfaObj.initialized, gpfaObj = gpfaObj.updateAll(); end

if ~exist('convergenceTol', 'var'), convergenceTol = 1e-6; end

if ~exist('startIter', 'var'), startIter = 1; end
numIters = (maxIters-startIter)+1;

FVEs = zeros(1, numIters);
Qs = zeros(1, numIters);
Hs = zeros(1, numIters);
tstart = tic;
converged = false;

allParams = {'R', 'C', 'D', 'b', 'taus', 'rhos', 'tauf', 'signs'};
lastParamValues = concatAllParams(gpfaObj, allParams);
lastDeltas = zeros(1, gpfaObj.kernel_update_freq);

mean_field_tolerance = 1e-3;
min_mean_field_tolerance = 1e-8;

idx = 1;
while idx < numIters
    itr = startIter+idx-1;

    % TODO (here or inside emStep): block updates of parameters that depend on each other
    [newObj, Qs(idx), Hs(idx)] = gpfaObj.emStep(itr, mean_field_tolerance);
    FVEs(idx) = gpfaObj.fve();
    elapsed = toc(tstart);
    
    newParamValues = concatAllParams(newObj, allParams);
    delta = norm(lastParamValues - newParamValues);
    lastDeltas = [lastDeltas(2:end) delta];
    if max(lastDeltas) < convergenceTol
        converged = true;
        break
    end
    
    if idx >= 2
        deltaLL = (Qs(idx)+Hs(idx))-(Qs(idx-1)+Hs(idx-1));
        deltaFVE = FVEs(idx) - FVEs(idx-1);
    else
        deltaLL = inf;
        deltaFVE = FVEs(idx);
    end
    
    fprintf('EM iteration %d/%d\tQ = %.2e\tdelta Q = %.2e\tdelta params = %.2e\tdelta FVE = %.2e\ttime per iteration = %.2fs\n', ...
        itr, maxIters, Qs(idx)+Hs(idx), deltaLL, delta, deltaFVE, elapsed / idx);

    if deltaLL < 0
        % DEBUGGING
        % changedParams = setdiff(allParams, gpfaObj.fixed);
        % for iParam=1:length(changedParams)
        %     tmpObj = prevObj;
        %     tmpObj.fixed = setdiff(allParams, changedParams(iParam));
        %     [newTmpObj, oldQ, oldH] = tmpObj.emStep(itr-1, mean_field_tolerance);
        %     [~, newQ, newH] = newTmpObj.emStep(itr, mean_field_tolerance);
        %     deltaParam = norm(concatAllParams(newTmpObj, allParams) - concatAllParams(tmpObj, allParams));
        %     fprintf('\t%s change by magnitude %.2e, delta LL = %.2e\n', [changedParams{iParam}], deltaParam, (newQ+newH)-(oldQ+oldH));
        % end
        % keyboard;
        if mean_field_tolerance > min_mean_field_tolerance
            mean_field_tolerance = mean_field_tolerance / 2;
            fprintf('\tmean_field_tolerance --> %.2e\n', mean_field_tolerance);
            continue
        else
            break
        end
    end
    
    % prevObj = gpfaObj; % DEBUGGING ONLY
    gpfaObj = newObj;
    lastParamValues = newParamValues;
    idx = idx + 1;
end

if ~converged
    warning('EM potentially exiting before convergence');
end

Qs = Qs(1:idx);
Hs = Hs(1:idx);

bestFit = gpfaObj;
[~, Qs(end+1), Hs(end+1)] = bestFit.emStep(maxIters, mean_field_tolerance);

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
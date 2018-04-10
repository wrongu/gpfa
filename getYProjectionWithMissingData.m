function P = getYProjectionWithMissingData(Y, diagD, C)
%GETYPROJECTIONWITHMISSINGDATA Compute a useful matrix product for the GPFA log likelihood when data
%Y contains missing (NaN) values. In terms of probabilistic models, missing data is simply omitted
%from the product over data likelihood. Translating this to matrix multiplications is not as
%straightforward as replacing NaN values with 0. 

% try
%     warning('Trying to compile mex file getYProjectionWithMissingData.c...');
%     mex getYProjectionWithMissingData.c
%     P = getYProjectionWithMissingData(Y, diagD, C);
% catch
%     error('Unable to compile mex file. Try running ''mex getYProjectionWithMissingData.c'' yourself and troubleshoot from there');
% end

Y(isnan(Y)) = 0;
P = Y * (C ./ diagD(:));
P = P(:);

end
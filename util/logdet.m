function ld = logdet(A)
% More stable log-determinant function. Copied from github.com:aecker/gpfa/lib/logdet.m
U = chol(A);
ld = 2 * sum(log(diag(U)));
end
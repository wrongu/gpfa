function d = logdet(X)

L = chol(X);
d = 2 * sum(log(diag(L)));

end
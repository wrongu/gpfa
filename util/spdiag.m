function spD = spdiag(D)
%SPDIAG create a sparse diagonal matrix with the diagonal created from elements of the vector D
%along the diagonal.

N = length(D);
spD = sparse(1:N, 1:N, D, N, N);

end
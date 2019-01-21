function val = tracedot(A, B)
% Effient trace(A' * B)
U = A.*B;
val = sum(U(:));
end
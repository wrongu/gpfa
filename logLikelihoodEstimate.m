function Q = logLikelihoodEstimate(Y, diagD, C, Kfull)

[mu, cov, G, P] = inferX(Y, diagD, C, Kfull);

Yz = Y;
Yz(isnan(Y)) = 0;

prior_term = -1/2 * (trace(Kfull * cov) + mu(:)'*Kfull*mu(:)) - 1/2 * logdet(Kfull);
likelihood_term = -1/2 * trace((Yz./diagD')*Yz') + mu(:)'*P - 1/2 * (trace(G * cov) + mu(:)' * G * mu(:));

Q = prior_term + likelihood_term;

end
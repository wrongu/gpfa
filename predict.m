function predY = predict(Y, diagD, C, Kfull)

[mu, ~] = inferX(Y, diagD, C, Kfull);
predY = mu * C';

end
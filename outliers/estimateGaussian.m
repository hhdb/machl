function [mu sigma2] = estimateGaussian(X)

[m, n] = size(X);
mu = zeros(n, 1);
sigma2 = zeros(n, 1);
mu = mean(X, 1);
sigma2 = var(X, opt=1, dim=1);

end
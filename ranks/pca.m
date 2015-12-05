function [U, S] = pca(X)

[m, n] = size(X);
U = zeros(n);
S = zeros(n);

sigma = (X' * X) / m;
[U, S, V] = svd(sigma);

end
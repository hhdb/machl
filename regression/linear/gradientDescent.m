function [theta, J_hist] = gradientDescent(X, y, theta, alpha, iters)

m = length(y);
J_hist = zeros(iters, 1);

for iter = 1:iters
    theta = theta - ((alpha / m) * (X' * (X * theta - y)));
    J_hist(iter) = computeCost(X, y, theta);
end

end
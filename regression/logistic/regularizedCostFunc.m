function [J, grad] = regularizedCostFunc(theta, X, y, lambda)

m = length(y);
grad = zeros(size(theta));
J = ((-y' * log(sigmoid(X * theta)) - (1 - y)' ...
	* log(1 - sigmoid(X * theta))) / m) ...
    + (lambda / (2 * m)) * sumsq(theta(2:size(theta)));
grad = ((X' * (sigmoid(X * theta) - y)) / m) + lambda / m * theta; 
grad(1) = ((X(:, 1)' * (sigmoid(X * theta) - y)) / m);

end
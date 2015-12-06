function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

X = reshape(params(1:num_movies*num_features), num_movies, num_features);
theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

X_grad = zeros(size(X));
theta_grad = zeros(size(theta));

J = sum(sum(((X * theta' - Y) .* R) .^ 2)) / 2
X_grad = (X * theta' - Y) .* R * theta;
theta_grad = ((X' * theta - Y') .* R') * X; 

J = J + ((sum(sum(theta .^ 2)) + sum(sum(X .^ 2))) * (lambda / 2));
X_grad = X_grad + lambda * X;
theta_grad = theta_grad + lambda * theta; 

grad = [X_grad(:); theta_grad(:)];

end
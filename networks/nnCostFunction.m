function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
J = 0;
theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));


a1 = [ones(m, 1), X];
z2 = a1 * theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2];
z3 = a2 * theta2';
a3=sigmoid(z3);

I = eye(num_labels); 
yy = I(:, y);   

cost = (-yy' .* log(a3)) - ((1 - yy)' .* log(1 - a3));
J = (sum(sum(cost))) / m;    

t1 = theta1;
t1(:, 1) = 0;
t2 = theta2;
t2(:, 1) = 0;
c = lambda ./ (2 .* m);
d = sum(sum(t1 .^ 2));
e = sum(sum(t2 .^ 2));
R = c .* (d + e);
J = J + R;


sigma3 = a3 - y;
sigma2 = (sigma3 * theta2 .* ...
         sigmoidGradient([ones(size(z2, 1), 1) z2]))(:, 2:end);
delta1 = sigma2' * a1;
delta2 = sigma3' * a2;
theta1_grad = delta1 ./ m + ...
            (lambda / m) * [zeros(size(theta1, 1), 1) theta1(:, 2:end)];
theta2_grad = delta2 ./ m + ...
            (lambda / m) * [zeros(size(theta2, 1), 1) theta2(:, 2:end)];


grad = [theta1_grad(:) ; theta2_grad(:)];

end
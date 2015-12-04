function p = predict(theta1, theta2, X)

m = size(X, 1);
num_labels = size(theta2, 1);
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * theta1');
h2 = sigmoid([ones(m, 1) h1] * theta2');
[p_val, p] = max(h2, [], 2);

end
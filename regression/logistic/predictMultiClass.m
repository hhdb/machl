function p = predictMultiClass(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];

prob = sigmoid(X * all_theta');
[maxProb, maxProbIdx] = max(prob, [], 2);
p = maxProbIdx(:);

end
function [all_theta] = multiClass(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);
all_theta = zeros(num_labels, n + 1);
X = [ones(m, 1) X];
options = optimset('GradObj', 'on', 'MaxIter', 50);

for c = 1:num_labels
  initial_theta = zeros(n + 1, 1);
  yBinaryClass = (y == c);

  [theta] = fmincg(@(t)(regularizedCostFunc(t, X, yBinaryClass, lambda)), ...
  	        initial_theta, options);
  all_theta(c, :) = theta(:)';
end

end
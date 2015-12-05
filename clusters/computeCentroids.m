function centroids = computeCentroids(X, idx, K)

[m n] = size(X);
centroids = zeros(K, n);

logicalidx = []; 
for i = 1:K 
	logicalidx = [logicalidx, [idx==i]]; 
end

centroids = X' * logicalidx;
num_points = sum(logicalidx, 1); 
num_points = repmat(num_points, n, 1);
centroids = centroids ./ num_points;
centroids = centroids';

end
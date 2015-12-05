function idx = findClosestCentroids(X, centroids)

K = size(centroids, 1);
idx = zeros(size(X,1), 1);

for i = 1:size(X) 
	for j = 1:K 
		distance(i, j) = sumsq(X(i, :) - centroids(j, :)); 
	end 
end

[min_value, row_I] = min(distance, [], 2); 
idx = row_I;

end
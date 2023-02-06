function centroids = computeCentroids(X, idx, K)

[m n] = size(X);

centroids = zeros(K, n);


for k=1:K
    centroids(k,:) = sum(X(find(idx == k),:),1) / size(find(idx == k),1);
end
end





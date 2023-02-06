function idx = findClosestCentroids(X, centroids)


idx = zeros(size(X,1), 1);


Z = zeros(size(X,1),size(centroids,1));

for i = 1:3
    Z(:,i) = sum((centroids(i,:) - X).^2 ,2);
end

[value,idx] = min(Z,[],2);


% =============================================================

end
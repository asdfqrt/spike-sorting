function Reduced_cluster = ReduceKmean(cluster, centroids, K, percent)


Reduced_cluster = zeros(ceil(size(cluster,1) * percent), K); %
Z = zeros(size(cluster,1), 1);

Z = sum((centroids - cluster).^2 ,2);
Y = [cluster Z];
Y = sortrows(Y, K+1);

Reduced_cluster = Y(1:ceil(size(cluster,1) * percent),1:K);

% =============================================================

end
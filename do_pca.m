function [U, S] = do_pca(X)


[m, n] = size(X);


U = zeros(n);
S = zeros(n);


sigma = 1/m * X' * X;
[U,S,V] = svd(sigma);



end
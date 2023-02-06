function g = sigmoidGradient(z)
g = zeros(size(z));

sig = sigmoid(z);
g= sig .* (1 - sig);
% g = (1 + sig) .* (1 - sig);







% =============================================================




end

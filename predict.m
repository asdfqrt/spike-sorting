function p = predict(Theta1, Theta2, Theta3, Theta4, Theta5, X)
m = size(X, 1);
num_labels = size(Theta5, 1);

p = zeros(size(X, 1), 1);


activation1 = [ones(m, 1) X];
z2 = activation1 * Theta1' ;
activation2 = sigmoid(z2);

activation2 = [ones(size(activation2, 1),1) activation2];
z3 = activation2 * Theta2';
activation3 = sigmoid(z3);

activation3 = [ones(size(activation3, 1),1) activation3];
z4 = activation3 * Theta3';
activation4 = sigmoid(z4);

activation4 = [ones(size(activation4, 1),1) activation4];
z5 = activation4 * Theta4';
activation5 = sigmoid(z5);

activation5 = [ones(size(activation5, 1),1) activation5];
z6 = activation5 * Theta5';
activation6 = sigmoid(z6);

[value,p] = max(activation6,[],2);



end
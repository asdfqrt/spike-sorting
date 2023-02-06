function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params(1 + size(Theta1(:)) : size(Theta1(:)) + hidden_layer_size * (hidden_layer_size +1)), ...
                 hidden_layer_size, (hidden_layer_size + 1));
             
Theta3 = reshape(nn_params(1 + size(Theta1(:)) + size(Theta2(:)):size(Theta1(:)) + size(Theta2(:)) + hidden_layer_size * (hidden_layer_size +1)), ...
                 hidden_layer_size, (hidden_layer_size + 1));
             
Theta4 = reshape(nn_params(1 + size(Theta1(:)) + size(Theta2(:)) + size(Theta3(:)):size(Theta1(:)) + size(Theta2(:)) + size(Theta3(:)) + hidden_layer_size * (hidden_layer_size +1)), ...
                 hidden_layer_size, (hidden_layer_size + 1));
             
Theta5 = reshape(nn_params(1 + size(Theta1(:)) + size(Theta2(:)) + size(Theta3(:)) + size(Theta4(:)):end), ...
                 num_labels, (hidden_layer_size + 1));
%              
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));
Theta4_grad = zeros(size(Theta4));
Theta5_grad = zeros(size(Theta5));

%% cost function%%


a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);

a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

a3 = [ones(m,1) a3];
z4 = a3 * Theta3';
a4 = sigmoid(z4);

a4 = [ones(m,1) a4];
z5 = a4 * Theta4';
a5 = sigmoid(z5);

a5 = [ones(m,1) a5];
z6 = a5 * Theta5';
a6 = sigmoid(z6);


h = a6;

y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
J = -1/m * sum(sum(y .* log(h) + (1-y) .* log(1 - h)));


%% regulation %%

regTheta1 = Theta1(:, 2:end);
regTheta2 = Theta2(:, 2:end);
regTheta3 = Theta3(:, 2:end);
regTheta4 = Theta4(:, 2:end);
regTheta5 = Theta5(:, 2:end);

regulation_term = (lambda / (2 * m)) * ( sum(sum(regTheta1 .^ 2)) + sum(sum(regTheta2 .^ 2)) + sum(sum(regTheta3 .^ 2)) + sum(sum(regTheta4 .^ 2)) + sum(sum(regTheta5 .^ 2)) );
J = J + regulation_term;
%% back propagation %%

DELTA1 = zeros(size(Theta1));
DELTA2 = zeros(size(Theta2));
DELTA3 = zeros(size(Theta3));
DELTA4 = zeros(size(Theta4));
DELTA5 = zeros(size(Theta5));


for t= 1:m
    a1_t = a1(t,:); %step1
    a2_t = a2(t, :);
    a3_t = a3(t, :);
    a4_t = a4(t, :);
    a5_t = a5(t, :);
    a6_t = a6(t, :);
    
    z2_t = [1; Theta1 * a1_t'];
    z3_t = [1; Theta2 * a2_t'];
    z4_t = [1; Theta3 * a3_t'];
    z5_t = [1; Theta4 * a4_t'];
    
    y_t = y(t, :);
    
    delta6 = a6_t - y_t;    %step2
    delta5 = Theta5' * delta6' .* sigmoidGradient(z5_t); %step3
    delta4 = Theta4' * delta5(2:end) .* sigmoidGradient(z4_t);
    delta3 = Theta3' * delta4(2:end) .* sigmoidGradient(z3_t);
    delta2 = Theta2' * delta3(2:end) .* sigmoidGradient(z2_t);
    
    
    DELTA1 = DELTA1 + delta2(2:end) * a1_t;
    DELTA2 = DELTA2 + delta3(2:end) * a2_t;
    DELTA3 = DELTA3 + delta4(2:end) * a3_t;
    DELTA4 = DELTA4 + delta5(2:end) * a4_t;
    DELTA5 = DELTA5 + delta6' * a5_t;
end



Theta1_grad = 1/m * DELTA1 + (lambda/m) * [zeros(size(Theta1, 1), 1) regTheta1];
Theta2_grad = 1/m * DELTA2 + (lambda/m) * [zeros(size(Theta2, 1), 1) regTheta2];
Theta3_grad = 1/m * DELTA3 + (lambda/m) * [zeros(size(Theta3, 1), 1) regTheta3];
Theta4_grad = 1/m * DELTA4 + (lambda/m) * [zeros(size(Theta4, 1), 1) regTheta4];
Theta5_grad = 1/m * DELTA5 + (lambda/m) * [zeros(size(Theta5, 1), 1) regTheta5];


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:) ; Theta4_grad(:) ; Theta5_grad(:) ;];

end

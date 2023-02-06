%% Initialization
clear ; close all; clc

%% ================== Load Example Dataset  ===================

fprintf('Loading data C_Difficult2_noise02\n\n');
load ('C_Difficult2_noise02.mat');

szspike = size(spike_times{1,1}, 2);
X = zeros(szspike,32);

for i =1:szspike
    for j = 1:32
        X(i,j) = data(1, spike_times{1,1}(1,i) + (j-1) );
    end
end



%% =================== Part 1: Dimension Reduction ===================

fprintf('plot PCA\n\n');

[U, S] = do_pca(X);

dimension = 3;

Z = projectData(X, U, dimension);
plot3(Z(:, 1), Z(:, 2),Z(:,3), 'b*');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 2: K-Means Clustering ======================

K = 3; % 3 Centroids
kmean_accuracy = 0;
new_kmean_accuracy = 0;

% repeat K_means
for i = 1:50
    initial_centroids = kMeansInitCentroids(Z, K);
    
    fprintf('\nRunning K-Means clustering on example dataset.\n\n');
    max_iters = 10;
    [new_centroids, new_idx] = runkMeans(Z, initial_centroids, max_iters, false);
    new_kmean_accuracy = mean(double(new_idx' == spike_class{1,1}));
    
    if new_kmean_accuracy > kmean_accuracy
        centroids = new_centroids;
        idx = new_idx;
        kmean_accuracy = new_kmean_accuracy;
    end
end

plotDataPoints(Z, idx, K);
fprintf('\nK-Means Done.\n\n');
fprintf('\nTraining Set Accuracy: %f\n', mean(double(idx' == spike_class{1,1})));
fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 3: K-Means 10% ======================

clu1_idx = idx == 1; % cluster1 빨강
clu2_idx = idx == 2; % cluster2 연두
clu3_idx = idx == 3; % cluster3 에메랄드
percent = 0.1;

red_clu1 = ReduceKmean(Z(clu1_idx,:), centroids(1,:), K, percent);
red_clu2 = ReduceKmean(Z(clu2_idx,:), centroids(2,:), K, percent);
red_clu3 = ReduceKmean(Z(clu3_idx,:), centroids(3,:), K, percent);

hold on;
plot3(red_clu1(:, 1), red_clu1(:, 2),red_clu1(:,3), 'b*'); %reduced_cluster1 파랑
plot3(red_clu2(:, 1), red_clu2(:, 2),red_clu2(:,3), 'r*'); %reduced_cluster2 빨강
plot3(red_clu3(:, 1), red_clu3(:, 2),red_clu3(:,3), 'y*'); %reduced_cluster3 노랑
hold off;


fprintf('\nReducing Done.\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%% Setup the parameters
X = [red_clu1; red_clu2; red_clu3];
y = [ones(size(red_clu1,1),1); 2*ones(size(red_clu2,1),1); 3*ones(size(red_clu3,1),1)];

input_layer_size  = size(X,2);  % 32 input nodes
hidden_layer_size = 256;   % 256 hidden units
num_labels = 3;          % 3 labels, from 1 to 3   
                  
%% ================ Part 4: Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta3 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta4 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta5 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:) ; initial_Theta4(:) ; initial_Theta5(:)];

%% =================== Part 5: Training NN ===================

fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 50);


lambda = 0.25;
tic
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);


[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


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
toc
fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 6: Implement Predict =================
tic
pred = predict(Theta1, Theta2, Theta3, Theta4, Theta5, Z);
toc
hold off;
plotDataPoints(Z, pred, K);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred' == spike_class{1,1})) * 100);
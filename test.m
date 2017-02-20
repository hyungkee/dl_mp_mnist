%% Initialization
%clear ; 
close all; clc

%% Setup the parameters
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 49;   %  hidden units
num_labels = 10;          % 10 labels, from 1 to 10
h = [hidden_layer_size];

%% Loading Data
fprintf('Loading Data ...\n')

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

m = 10000;
X = images(:,1:m); % 784 x m

y = labels(1:m,:)'; % 1 x 60000
y(y == 0) = 10; % 0 is replaced to 10
Y = zeros(10, m); % 10 x m
for t=1:m
    Y(y(t),t) = 1;
end

%% Training NN
fprintf('\nTraining Neural Network... \n')

options.active = 'ReLU';
options.MaxIter = 100;
options.epsilon_init = 0.3;
options.eta = 2;
options.dG = 0.003;

[model, mse] = mlp(X, Y, h, options);


%% Implement Prediction
pred_Y = mlpPred(model, X);
[~, pred_y] = max(pred_Y, [], 1);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_y == y)) * 100);

%% apply to test set
images = loadMNISTImages('t10k-images.idx3-ubyte');
labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

test_X = images;
test_y = labels';
test_y(test_y == 0) = 10; % 0 is replaced to 10

pred_test_Y = mlpPred(model, test_X);
[~, pred_test_y] = max(pred_test_Y, [], 1);

fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred_test_y == test_y)) * 100);

%% test_knn_classifier.m
% This script trains and evaluates a K-NN classifier for pedestrian detection.
% It loads the dataset, extracts features, splits into training and testing sets,
% trains K-NN with different values of K, evaluates performance, and visualizes results.

clear; clc; close all;

%% Load and prepare dataset
fprintf('Loading dataset...\n');
cfg = config();
[X, Y] = load_dataset(cfg);

% Keep a copy of the original file paths or image data for visualization later
originalFiles = X;

fprintf('Splitting dataset (70/30)...\n');
% Split the dataset into 70% training and 30% testing
[trainIdx, testIdx] = split_dataset(Y, 0.3, 42);

%% Extract gradient-based features (HOG-like features)
% Each image is converted to grayscale, resized, and its gradients are used
% to form a simple descriptor similar to HOG. This does not require any toolbox.

cellSize = [8 8];
features = [];

for i = 1:length(X)
    img = X{i};

    % Read image if X contains file paths instead of image data
    if ischar(img) || isstring(img)
        img = imread(img);
    end

    % Convert to grayscale if the image is in color
    if size(img,3) > 1
        img = rgb2gray(img);
    end

    % Resize to a standard pedestrian window size
    img = imresize(img, [128 64]);

    % Compute simple gradient-based magnitude feature
    Gx = imfilter(double(img), [-1 0 1], 'replicate');
    Gy = imfilter(double(img), [-1; 0; 1], 'replicate');
    mag = sqrt(Gx.^2 + Gy.^2);
    feat = imresize(mag, [16 8]);
    feat = feat(:)'; % Flatten into a row vector

    features = [features; feat];
end

X = double(features); % Convert to numeric matrix for training

%% Split features into training and testing sets
XTrain = X(trainIdx, :);
YTrain = Y(trainIdx);
XTest  = X(testIdx, :);
YTest  = Y(testIdx);

%% Train and evaluate K-NN classifier for multiple K values
K_values = [1 3 5 7 9];
accuracy = zeros(size(K_values));

for i = 1:length(K_values)
    model = KNN_classifier(XTrain, YTrain, K_values(i));
    YPred = predict(model, XTest);
    acc = sum(YPred == YTest) / numel(YTest);
    accuracy(i) = acc;
    fprintf('K = %d --> Accuracy = %.2f%%\n', K_values(i), acc*100);
end

%% Plot accuracy against K
figure;
plot(K_values, accuracy*100, '-o', 'LineWidth', 1.5);
xlabel('K (Number of Neighbours)');
ylabel('Accuracy (%)');
title('K-NN Classification Performance');
grid on;

%% Evaluate the best model (highest accuracy)
[~, idx] = max(accuracy);
bestK = K_values(idx);
fprintf('\nBest K = %d\n', bestK);

model = KNN_classifier(XTrain, YTrain, bestK);
YPred = predict(model, XTest);
cm = confusionmat(YTest, YPred);

%% Display confusion matrix
figure;
confusionchart(cm, {'Non-Pedestrian','Pedestrian'});
title(sprintf('Confusion Matrix (K = %d)', bestK));

%% Compute and print evaluation metrics
TP = cm(2,2); FP = cm(1,2);
FN = cm(2,1); TN = cm(1,1);

precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1 = 2 * (precision * recall) / (precision + recall);

fprintf('\nPrecision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1);
fprintf('Accuracy: %.2f%%\n', 100*accuracy(idx));

% Save results for documentation
save('knn_results.mat','K_values','accuracy','bestK','precision','recall','f1');

fprintf('\nKNN evaluation complete.\n');

%% Preview random test predictions
% This section shows a few random test images with predicted and true labels
testIds = find(testIdx);                % numeric indices of test samples
numShow = min(6, numel(testIds));       % number of images to show
pickAbs = randsample(testIds, numShow); % absolute indices of samples
[~, pickInTest] = ismember(pickAbs, testIds);

figure;
for i = 1:numShow
    src = originalFiles{pickAbs(i)};
    if ischar(src) || isstring(src)
        img = imread(src);
    else
        img = src;
    end
    subplot(2,3,i);
    imshow(img);
    title(sprintf('Pred: %d | True: %d', YPred(pickInTest(i)), YTest(pickInTest(i))));
end
sgtitle('Sample Test Predictions');

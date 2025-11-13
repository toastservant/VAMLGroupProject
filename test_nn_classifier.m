% Train/test the nearest-neighbour pedestrian classifier using the same
% preprocessing + HOG pipeline and dataset split as the kNN and SVM script.

clear; clc; close all;

cfg = config();
testRatio = 0.3; % 70/30 split like SVM evaluation

fprintf('Loading dataset...\n');
[fileList, labels] = load_dataset(cfg);
if isempty(fileList)
    error('test_nn_classifier:EmptyDataset', 'No samples found in crops directories.');
end

fprintf('Extracting HOG features from %d samples...\n', numel(fileList));
features = build_feature_matrix(fileList, cfg);

fprintf('Splitting dataset (%.0f%% train / %.0f%% test)...\n', ...
    (1 - testRatio) * 100, testRatio * 100);
[trainIdx, testIdx] = split_dataset(labels, testRatio, cfg.seed);

trainImages = features(trainIdx, :);
trainLabels = labels(trainIdx);
testImages  = features(testIdx, :);
testLabels  = labels(testIdx);

modelNN = NNtraining(trainImages, trainLabels);

% Predict labels for test set
predictedLabels = zeros(sum(testIdx), 1);
for i = 1:size(testImages, 1)
    predictedLabels(i) = NN_classifier(testImages(i, :), modelNN);
end

accuracy = mean(predictedLabels == testLabels);
fprintf('1-NN accuracy: %.2f%%\n', accuracy * 100);

% Metrics matching the SVM script
cm = confusionmat(testLabels, predictedLabels, 'Order', [-1 1]);
TN = cm(1,1); FP = cm(1,2);
FN = cm(2,1); TP = cm(2,2);
precision = TP / max(1, (TP + FP));
recall = TP / max(1, (TP + FN));
f1 = 2 * precision * recall / max(eps, precision + recall);

fprintf('Precision: %.2f | Recall: %.2f | F1: %.2f\n', precision, recall, f1);

figure;
confusionchart(cm, {'Non-Pedestrian','Pedestrian'});
title('NN Confusion Matrix (k=1)');

results = struct('cfg', cfg, 'modelNN', modelNN, ...
    'trainIdx', trainIdx, 'testIdx', testIdx, ...
    'trainImages', trainImages, 'trainLabels', trainLabels, ...
    'testImages', testImages, 'testLabels', testLabels, ...
    'predictedLabels', predictedLabels, ...
    'confusionMatrix', cm, ...
    'metrics', struct('accuracy', accuracy, 'precision', precision, 'recall', recall, 'f1', f1));

save('nn_results.mat', 'results', 'features', 'labels', 'fileList');
fprintf('Saved evaluation details to nn_results.mat (features + results).\n');

%% ------------------------------------------------------------------------
function features = build_feature_matrix(files, cfg)
numSamples = numel(files);
sampleFeat = extract_hog(preprocess_image(imread(files{1}), cfg), cfg);
featLen = numel(sampleFeat);
features = zeros(numSamples, featLen, 'double');
features(1, :) = sampleFeat;

for i = 2:numSamples
    Iproc = preprocess_image(imread(files{i}), cfg);
    features(i, :) = extract_hog(Iproc, cfg);
end
end

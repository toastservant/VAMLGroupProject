% Train/evaluate an SVM pedestrian classifier using preprocess_image + HOG features.

clear; clc; close all;

cfg = config();
testRatio = 0.3;
svmOpts = struct('kernel', 'linear', 'boxConstraint', 1, 'kernelScale', 'auto');
rng(cfg.seed);

fprintf('Loading dataset...\n');
[files, labels] = load_dataset(cfg);
if isempty(files)
    error('test_svm_classifier:EmptyDataset', 'No samples found in crops directory.');
end

fprintf('Extracting HOG features from %d samples...\n', numel(files));
features = compute_hog_features(files, cfg);

fprintf('Splitting dataset (%.0f%% train / %.0f%% test)...\n', ...
    (1 - testRatio) * 100, testRatio * 100);
[trainIdx, testIdx] = split_dataset(labels, testRatio, cfg.seed);

XTrain = features(trainIdx, :);
YTrain = labels(trainIdx);
XTest = features(testIdx, :);
YTest = labels(testIdx);

model = svm_classifier(XTrain, YTrain, svmOpts);
[YPred, scoresRaw] = predict(model, XTest);
if size(scoresRaw, 2) >= 2
    scores = scoresRaw(:, 2);
else
    scores = scoresRaw;
end
accuracy = mean(YPred == YTest);
fprintf('SVM accuracy: %.2f%%\n', accuracy * 100);

cm = confusionmat(YTest, YPred, 'Order', [-1 1]);
TN = cm(1,1); FP = cm(1,2);
FN = cm(2,1); TP = cm(2,2);
precision = TP / max(1, (TP + FP));
recall = TP / max(1, (TP + FN));
f1 = 2 * precision * recall / max(eps, precision + recall);

fprintf('Precision: %.2f | Recall: %.2f | F1: %.2f\n', precision, recall, f1);

figure;
confusionchart(cm, {'Non-Pedestrian','Pedestrian'});
title(sprintf('SVM Confusion Matrix (kernel=%s)', svmOpts.kernel));

% Preview a few test predictions with confidence
testIds = find(testIdx);
numShow = min(6, numel(testIds));
if numShow > 0
    pickIdx = randperm(numel(testIds), numShow);
    pickAbs = testIds(pickIdx);
    figure;
    for i = 1:numShow
        absIdx = pickAbs(i);
        img = imread(files{absIdx});
        testPos = pickIdx(i);
        subplot(2,3,i);
        imshow(img);
        cls = YPred(testPos);
        conf = scores(testPos);
        title(sprintf('Pred: %d (%.2f)\nTrue: %d', cls, conf, YTest(testPos)));
    end
    sgtitle('Sample Test Predictions');
end

results = struct('cfg', cfg, 'svmOpts', svmOpts, 'model', model, ...
    'trainIdx', trainIdx, 'testIdx', testIdx, ...
    'XTrain', XTrain, 'YTrain', YTrain, ...
    'XTest', XTest, 'YTest', YTest, ...
    'YPred', YPred, 'scores', scores, ...
    'confusionMatrix', cm, ...
    'metrics', struct('accuracy', accuracy, 'precision', precision, 'recall', recall, 'f1', f1));

save('svm_results.mat', 'results');

fprintf('Evaluation complete. Saved results to svm_results.mat\n');

%% ------------------------------------------------------------------------
function features = compute_hog_features(files, cfg)
% compute_hog_features Apply preprocess_image + extract_hog per file
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

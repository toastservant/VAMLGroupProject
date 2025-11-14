%% compare_svm_kernels.m
% Compare linear vs. RBF vs. polynomial kernels using the baseline HOG descriptor.

clear; clc; close all;

cfg = config();
% Use preferred preprocessing combo
cfg.equalizeHist = false;
cfg.denoise = true;
testRatio = 0.3;

fprintf('Loading dataset...\n');
[fileList, labels] = load_dataset(cfg);
if isempty(fileList)
    error('compare_svm_kernels:EmptyDataset', 'No crops available.');
end

fprintf('Building baseline HOG feature matrix...\n');
features = build_feature_matrix(fileList, cfg);

fprintf('Splitting dataset (%.0f%% train / %.0f%% test)...\n', ...
    (1 - testRatio) * 100, testRatio * 100);
[trainIdx, testIdx] = split_dataset(labels, testRatio, cfg.seed);
XTrain = features(trainIdx, :);
YTrain = labels(trainIdx);
XTest  = features(testIdx, :);
YTest  = labels(testIdx);

kerConfigs = {
    'Linear', struct('kernel', 'linear', 'kernelScale', 'auto');
    'RBF',    struct('kernel', 'rbf',    'kernelScale', 'auto');
    'Poly2',  struct('kernel', 'polynomial', 'polynomialOrder', 2)
};

results = struct('Name', [], 'Accuracy', [], 'Precision', [], 'Recall', [], 'F1', []);

for i = 1:size(kerConfigs, 1)
    name = kerConfigs{i, 1};
    opts = kerConfigs{i, 2};
    fprintf('\n[%s] Training SVM...\n', name);
    model = svm_classifier(XTrain, YTrain, opts);
[YPred, ~] = predict(model, XTest);

    cm = confusionmat(YTest, YPred, 'Order', [-1 1]);
    TN = cm(1,1); FP = cm(1,2);
    FN = cm(2,1); TP = cm(2,2);
    accuracy = mean(YPred == YTest);
    precision = TP / max(1, (TP + FP));
    recall = TP / max(1, (TP + FN));
f1 = 2 * precision * recall / max(eps, precision + recall);

    results(i).Name = name;
    results(i).Accuracy = accuracy;
    results(i).Precision = precision;
    results(i).Recall = recall;
    results(i).F1 = f1;

    fprintf('[%s] Accuracy: %.2f%% | Precision: %.2f | Recall: %.2f | F1: %.2f\n', ...
        name, accuracy * 100, precision, recall, f1);
end

comparisonTable = struct2table(results);
disp('SVM kernel comparison:');
disp(comparisonTable);

figure;
metricsMatrix = [[results.Accuracy]' [results.Precision]' [results.Recall]' [results.F1]'];
bar(metricsMatrix * 100);
set(gca, 'XTickLabel', {results.Name});
legend({'Accuracy %','Precision %','Recall %','F1 %'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
ylabel('Percentage');
title('SVM Kernel Comparison');
grid on;
ylim([90 100]);

save('svm_kernel_comparison.mat', 'comparisonTable', 'results', 'kerConfigs', 'cfg', 'trainIdx', 'testIdx');
fprintf('Stored kernel comparison results in svm_kernel_comparison.mat\n');

%% ------------------------------------------------------------------------
function features = build_feature_matrix(files, cfg)
numSamples = numel(files);
sampleFeat = extract_hog(preprocess_image(imread(files{1}), cfg), cfg);
featLen = numel(sampleFeat);
features = zeros(numSamples, featLen, 'double');
features(1, :) = sampleFeat;
for i = 2:numSamples
    features(i, :) = extract_hog(preprocess_image(imread(files{i}), cfg), cfg);
end
end

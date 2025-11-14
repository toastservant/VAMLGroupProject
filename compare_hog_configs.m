%% compare_hog_configs.m
% Compare baseline vs. dense HOG descriptors using the SVM pipeline (Prac 5).

clear; clc; close all;

cfg = config();
% Use the preprocessing combo found best in experiments
cfg.equalizeHist = false;
cfg.denoise = true;
testRatio = 0.3;

fprintf('Loading dataset...\n');
[fileList, labels] = load_dataset(cfg);
if isempty(fileList)
    error('compare_hog_configs:EmptyDataset', 'No crops available for comparison.');
end

fprintf('Preparing deterministic train/test split...\n');
[trainIdx, testIdx] = split_dataset(labels, testRatio, cfg.seed);

experiments = {
    'BaselineHOG', @extract_hog,            'CellSize [8 8], Block [2 2], Overlap [1 1]';
    'DenseHOG',    @dense_hog_descriptor,   'CellSize [4 4], Block [2 2], Overlap [1 1]'
};

results = struct('Name', [], 'Description', [], 'Accuracy', [], ...
    'Precision', [], 'Recall', [], 'F1', []);

for i = 1:size(experiments, 1)
    expName = experiments{i, 1};
    hogFunc = experiments{i, 2};
    desc = experiments{i, 3};

    fprintf('\n[%s] Building feature matrix...\n', expName);
    feats = build_feature_matrix(fileList, cfg, hogFunc);

    XTrain = feats(trainIdx, :);
    YTrain = labels(trainIdx);
    XTest = feats(testIdx, :);
    YTest = labels(testIdx);

    svmOpts = struct('kernel', 'linear', 'standardize', true);
    model = svm_classifier(XTrain, YTrain, svmOpts);
    YPred = predict(model, XTest);

    cm = confusionmat(YTest, YPred, 'Order', [-1 1]);
    TN = cm(1,1); FP = cm(1,2);
    FN = cm(2,1); TP = cm(2,2);
    accuracy = mean(YPred == YTest);
    precision = TP / max(1, (TP + FP));
    recall = TP / max(1, (TP + FN));
    f1 = 2 * precision * recall / max(eps, precision + recall);

    results(i).Name = expName;
    results(i).Description = desc;
    results(i).Accuracy = accuracy;
    results(i).Precision = precision;
    results(i).Recall = recall;
    results(i).F1 = f1;

    fprintf('[%s] Accuracy: %.2f%% | Precision: %.2f | Recall: %.2f | F1: %.2f\n', ...
        expName, accuracy * 100, precision, recall, f1);
end

comparisonTable = struct2table(results);
disp('HOG configuration comparison:');
disp(comparisonTable);

figure;
metricsMatrix = [[results.Accuracy]' [results.Precision]' [results.Recall]' [results.F1]'];
bar(metricsMatrix * 100);
set(gca, 'XTickLabel', {results.Name});
legend({'Accuracy %','Precision %','Recall %','F1 %'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
ylabel('Percentage');
title('Baseline vs Dense HOG Performance');
grid on;
ylim([90 100]);

save('hog_comparison.mat', 'comparisonTable', 'results', 'experiments', 'cfg', 'trainIdx', 'testIdx');
fprintf('Stored comparison results in hog_comparison.mat\n');

%% ------------------------------------------------------------------------
function feats = build_feature_matrix(files, cfg, hogFunc)
numSamples = numel(files);
sampleFeat = hog_from_image(files{1}, cfg, hogFunc);
featLen = numel(sampleFeat);
feats = zeros(numSamples, featLen, 'double');
feats(1, :) = sampleFeat;
for i = 2:numSamples
    feats(i, :) = hog_from_image(files{i}, cfg, hogFunc);
end
end

function feat = hog_from_image(path, cfg, hogFunc)
Iin = imread(path);
Iproc = preprocess_image(Iin, cfg);
feat = hogFunc(Iproc, cfg);
end

function feat = dense_hog_descriptor(Iproc, cfg)
if ~isequal(size(Iproc, 1), cfg.targetSize(1)) || ~isequal(size(Iproc, 2), cfg.targetSize(2))
    Iproc = imresize(Iproc, cfg.targetSize);
end
feat = extractHOGFeatures(Iproc, ...
    'CellSize', [4 4], ...
    'BlockSize', [2 2], ...
    'BlockOverlap', [1 1], ...
    'NumBins', 9);
feat = double(feat(:)');
end

% Preview preprocessing pipeline before training
cfg = config;
[files, labels] = load_dataset(cfg);


% mumbo jumbo to get a random sample of images used for testing
rng(cfg.seed); % repeatable sampling
posIdx = find(labels == 1);
negIdx = find(labels == -1);

numPos = min(10, numel(posIdx));
numNeg = min(10, numel(negIdx));

posSample = posIdx(randperm(numel(posIdx), numPos));
negSample = negIdx(randperm(numel(negIdx), numNeg));
sampleIdx = [posSample(:); negSample(:)];

% put test samples in a list and remove from orignal files list
testSample = [files(posSample), files(negSample)];
testSample = testSample(:);

[~, idxToRemove] = ismember(testSample, files);
testlabels = labels(idxToRemove(idxToRemove > 0));

files(idxToRemove(idxToRemove > 0)) = [];
labels(idxToRemove(idxToRemove > 0)) = [];


% pre process all images and hog it
trainImages = [];
testImages = [];
for i = 1:numel(files)
    Iin = imread(files{i});
    Iout = preprocess_image(Iin, cfg);
    hog = hog_feature_vector(Iout);
    trainImages = [trainImages; hog];
end

modelNN = NNtraining(trainImages, labels);

for i = 1:numel(testSample)
    Iin = imread(testSample{i});
    Iout = preprocess_image(Iin, cfg);
    hog = hog_feature_vector(Iout);
    testImages = [testImages; hog];
end

% test sample images with trained images
for i = 1:size(testImages,1)
    testnumber = testImages(i,:);
    classificationResult(i,1) = NN_classifier(testnumber, modelNN);

end

% evaluate results

% Finally we compared the predicted classification from our mahcine
% learning algorithm against the real labelling of the testing image
comparison = (testlabels==classificationResult);

%Accuracy is the most common metric. It is defiend as the numebr of
%correctly classified samples/ the total number of tested samples
Accuracy = sum(comparison)/length(comparison)

confMatrix = confusionmat(testlabels, classificationResult);

% Display the confusion matrix
disp('Confusion Matrix:');
disp(confMatrix);

% Create a heatmap with custom labels
figure;
h = heatmap(confMatrix);

% Set the axis labels to your class labels
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';
h.Title = 'Confusion Matrix';

% Change the displayed tick labels to 0 and 1
h.XDisplayLabels = {'Non Pedestrian','Pedestrian'};
h.YDisplayLabels = {'Non Pedestrian','Pedestrian'};

TP = confMatrix(1,1);  % True Positives
TN = confMatrix(2,2);  % True Negatives
FP = confMatrix(1,2);  % False Positives
FN = confMatrix(2,1);  % False Negatives

% Precision
precision = TP / (TP + FP);
% Recall (Sensitivity)
recall = TP / (TP + FN);
% F1-Score
F1 = 2 * (precision * recall) / (precision + recall);

fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1-Score: %.2f\n', F1);
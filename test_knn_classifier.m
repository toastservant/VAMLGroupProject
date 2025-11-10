clear; clc; close all;

% Load and preprocess dataset
fprintf('Loading dataset...\n');
[X, Y] = load_dataset();

fprintf('Splitting dataset (70/30)...\n');
[trainIdx, testIdx] = split_dataset(Y, 0.3);
XTrain = X(trainIdx, :);
YTrain = Y(trainIdx);
XTest  = X(testIdx, :);
YTest  = Y(testIdx);

% Test multiple K values
K_values = [1 3 5 7 9];
accuracy = zeros(size(K_values));

for i = 1:length(K_values)
    model = KNN_classifier(XTrain, YTrain, K_values(i));
    YPred = predict(model, XTest);
    acc = sum(YPred == YTest) / numel(YTest);
    accuracy(i) = acc;
    fprintf('K = %d --> Accuracy = %.2f%%\n', K_values(i), acc*100);
end

% Plot K vs Accuracy
figure;
plot(K_values, accuracy*100, '-o', 'LineWidth', 1.5);
xlabel('K (Number of Neighbours)');
ylabel('Accuracy (%)');
title('KNN Classification Performance');
grid on;

% Confusion Matrix for Best K
[~, idx] = max(accuracy);
bestK = K_values(idx);
fprintf('\nBest K = %d\n', bestK);

model = KNN_classifier(XTrain, YTrain, bestK);
YPred = predict(model, XTest);
cm = confusionmat(YTest, YPred);

figure;
confusionchart(cm, {'Non-Pedestrian','Pedestrian'});
title(sprintf('Confusion Matrix (K = %d)', bestK));

% Compute metrics
TP = cm(2,2); FP = cm(1,2);
FN = cm(2,1); TN = cm(1,1);
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1 = 2*(precision*recall)/(precision+recall);

fprintf('\nPrecision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1);
fprintf('Accuracy: %.2f%%\n', accuracy(idx)*100);

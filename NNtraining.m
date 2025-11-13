function modelNN = NNtraining(images, labels)
% NNtraining Simple wrapper to store training samples for 1-NN baseline.
modelNN.neighbours = images;
modelNN.labels = labels;
end

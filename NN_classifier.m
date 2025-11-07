function nnC = NN_classifier(inputImg, modelNN)
    predict = inf;
    for i=1:size(modelNN.neighbours,1)
        dEuc = EuclideanDistance(inputImg,modelNN.neighbours(i,:));
        if dEuc < predict
            predict = dEuc; % Update prediction with the new minimum distance
            nnC = modelNN.labels(i);
        end
    end
end

function dEuc=EuclideanDistance(sample1, sample2)
    sum = 0;
    for i=1:size(sample1,2)
        sum = sum + (sample1(1,i) - sample2(1,i))^2;
    end 
dEuc = sqrt(sum);
%dEuc = sqrt(sum((sample1 - sample2).^2));
end

function modelNN = NNtraining(images, labels)

modelNN.neighbours=images;
modelNN.labels=labels;

end
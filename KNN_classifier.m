
function model = KNN_classifier(XTrain, YTrain, k)
    if nargin < 3
        k = 5; % default number of neighbours
    end
    
    fprintf('Training KNN classifier with K = %d...\n', k);
    model = fitcknn(XTrain, YTrain, ...
        'NumNeighbors', k, ...
        'Distance', 'euclidean', ...
        'Standardize', true);
    
    fprintf('Training complete.\n');
end

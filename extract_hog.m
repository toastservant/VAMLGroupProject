function f = extract_hog(I, cfg)
% extract_hog Compute pedestrian HOG descriptor 
% I should already be a grayscale crop from preprocess_image. Practical 5
% pipeline is: reshape/resize -> HOG -> row vector for the classifier.

arguments
    I {mustBeNumeric}
    cfg struct
end

targetSize = cfg.targetSize;
hogParams = { ...
    'CellSize', [8 8], ...
    'BlockSize', [2 2], ...
    'BlockOverlap', [1 1], ...
    'NumBins', 9 ...
};

if ~isequal(size(I, 1), targetSize(1)) || ~isequal(size(I, 2), targetSize(2))
    I = imresize(I, targetSize);
end

f = extractHOGFeatures(I, hogParams{:});
f = double(f(:)'); % ensure 1xD row vector for feature matrix

% Optional helper for debugging: call visualize_hog(I, cfg)
end

function visualize_hog(I, cfg)
% visualize_hog Show HOG either via showHog (Prac 5) or MATLAB fallback
if ~isequal(size(I, 1), cfg.targetSize(1)) || ~isequal(size(I, 2), cfg.targetSize(2))
    I = imresize(I, cfg.targetSize);
end

if exist('showHog', 'file')
    descriptor = extract_hog(I, cfg);
    showHog(descriptor, cfg.targetSize);
    title('HOG visualization (showHog)');
else
    [~, hogVis] = extractHOGFeatures(I, ...
        'CellSize', [8 8], ...
        'BlockSize', [2 2], ...
        'BlockOverlap', [1 1], ...
        'NumBins', 9, ...
        'Visualization', true);
    imshow(I, []);
    hold on;
    plot(hogVis);
    title('HOG visualization (extractHOGFeatures)');
    hold off;
end
end

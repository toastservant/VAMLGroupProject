% Preview HOG feature extraction pipeline (Prac 5 Task 1)
cfg = config;
[files, labels] = load_dataset(cfg);

rng(cfg.seed); % repeatable sampling aligned with preview_preprocessing
posIdx = find(labels == 1);
negIdx = find(labels == -1);

numPos = min(3, numel(posIdx));
numNeg = min(3, numel(negIdx));

if numPos < 3
    warning('preview_hog:NotEnoughPos', 'Only %d positive samples available.', numPos);
end
if numNeg < 3
    warning('preview_hog:NotEnoughNeg', 'Only %d negative samples available.', numNeg);
end

if numPos > 0
    posSample = posIdx(randperm(numel(posIdx), numPos));
else
    posSample = [];
end

if numNeg > 0
    negSample = negIdx(randperm(numel(negIdx), numNeg));
else
    negSample = [];
end
sampleIdx = [posSample(:); negSample(:)];

if isempty(sampleIdx)
    error('preview_hog:NoSamples', 'No samples available to preview.');
end

numCols = numel(sampleIdx);
figure('Name', 'HOG Preview');
posCount = 0;
negCount = 0;
hogLength = [];
hogParams = { ...
    'CellSize', [8 8], ...
    'BlockSize', [2 2], ...
    'BlockOverlap', [1 1], ...
    'NumBins', 9 ...
};

for i = 1:numCols
    idx = sampleIdx(i);
    Iin = imread(files{idx});
    Iproc = preprocess_image(Iin, cfg);
    hogFeat = extract_hog(Iproc, cfg);
    hogLength(end + 1) = numel(hogFeat); %#ok<SAGROW>

    isPositive = labels(idx) > 0;
    if isPositive
        posCount = posCount + 1;
        originalTitle = sprintf('Original +ve %d', posCount);
        processedTitle = sprintf('Processed +ve %d', posCount);
        hogTitle = sprintf('HOG +ve %d', posCount);
    else
        negCount = negCount + 1;
        originalTitle = sprintf('Original -ve %d', negCount);
        processedTitle = sprintf('Processed -ve %d', negCount);
        hogTitle = sprintf('HOG -ve %d', negCount);
    end

    subplot(3, numCols, i);
    imshow(Iin);
    title(originalTitle);

    subplot(3, numCols, numCols + i);
    imshow(Iproc);
    title(processedTitle);

    ax = subplot(3, numCols, 2 * numCols + i);
    show_hog_subplot(ax, Iproc, hogFeat, cfg, hogTitle, hogParams);
end

if ~isempty(hogLength)
    fprintf('HOG feature length: %d\n', hogLength(1));
end

function show_hog_subplot(ax, Iproc, hogFeat, cfg, titleStr, hogParams)
% show_hog_subplot Visualise HOG using showHog (if available) or MATLAB fallback
axes(ax);
cla(ax);

if exist('showHog', 'file') == 2
    showHog(hogFeat, cfg.targetSize);
    axis tight off;
else
    [~, hogVis] = extractHOGFeatures(Iproc, hogParams{:});
    imshow(Iproc, 'Parent', ax);
    hold(ax, 'on');
    plot(hogVis);
    hold(ax, 'off');
    axis(ax, 'tight');
end

title(ax, titleStr);
end

% Preview preprocessing pipeline before training
cfg = config;
[files, labels] = load_dataset(cfg);

rng(cfg.seed); % repeatable sampling
posIdx = find(labels == 1);
negIdx = find(labels == -1);

numPos = min(3, numel(posIdx));
numNeg = min(3, numel(negIdx));

if numPos < 3
    warning('preview_preprocessing:NotEnoughPos', 'Only %d positive samples available.', numPos);
end
if numNeg < 3
    warning('preview_preprocessing:NotEnoughNeg', 'Only %d negative samples available.', numNeg);
end

posSample = posIdx(randperm(numel(posIdx), numPos));
negSample = negIdx(randperm(numel(negIdx), numNeg));
sampleIdx = [posSample(:); negSample(:)];

figure('Name', 'Preprocessing Preview');
posCount = 0;
negCount = 0;
lastProcessed = [];

for i = 1:numel(sampleIdx)
    idx = sampleIdx(i);
    Iin = imread(files{idx});
    Iout = preprocess_image(Iin, cfg);
    lastProcessed = Iout;

    isPositive = labels(idx) > 0;
    if isPositive
        posCount = posCount + 1;
        originalTitle = sprintf('Original +ve %d', posCount);
        processedTitle = sprintf('Processed +ve %d', posCount);
    else
        negCount = negCount + 1;
        originalTitle = sprintf('Original -ve %d', negCount);
        processedTitle = sprintf('Processed -ve %d', negCount);
    end

    subplot(2, 6, i);
    imshow(Iin);
    title(originalTitle);

    subplot(2, 6, i + 6);
    imshow(Iout);
    title(processedTitle);
end

if ~isempty(lastProcessed)
    sz = size(lastProcessed);
    fprintf('Processed image size: [%d %d]\n', sz(1), sz(2));
end

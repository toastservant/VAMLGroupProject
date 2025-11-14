% Multi-scale pedestrian detemaximctor demo using preprocess_image-style pipeline + HOG features.

clear; clc; close all;

cfg = config();
modelFile = 'svm_results.mat';
imageDir = fullfile(pwd, 'images');
gtFile = fullfile(pwd, 'test.dataset');
drawGT = true; % overlay red boxes from test.dataset if available
params.baseStride = [24 24];        % default stride before scale adjustments
params.scoreThreshold = 0.7;        % minimum score after penalties (tuned for far recall)
params.nmsThreshold = 0.15;         % tighter IoU threshold for non-max suppression
params.aspectTolerance = 0.2;       % tighter aspect ratio (reject trees/poles)
params.gradientThreshold = 0.05;  % base gradient energy threshold
params.scales = [1.35 1.15 1.0 0.9 0.75 0.6 0.5 0.4 0.32 0.24 0.18];
params.scalePenalties = [0.4 0.25 0.05 0 0 0.02 0.05 0.08 0.08 0.1 0.1];
params.scaleStrideFactors = [1 1 1 1.1 1.3 1.6 1.9 2.2 2.6 3.0 3.4];
params.scaleGradAdjust = [0 0 0 0 -0.01 -0.015 -0.02 -0.025 -0.035 -0.04 -0.05];
params.maxLongSide = 900;
params.maxDetectionsToShow = 25;
params.nestedContainment = 0.85;                    % drop boxes fully inside stronger ones
params.lowScoreCutoff = 0.75;                       % if score below this AND box is skinny -> drop
params.minWidthForLowScore = 50;                    % width threshold (pixels in original coordinates)
params.minAspectForLowScore = 0.35;                 % drop tall skinny boxes unless high score
params.maxAspectForLowScore = 0.9;                  % drop extra-wide boxes unless high score
params.highScoreOverride = 0.9;                     % skip aspect penalties when score >= this

assert(numel(params.scales) == numel(params.scalePenalties), ...
    'scales and scalePenalties must have same length.');
assert(numel(params.scales) == numel(params.scaleStrideFactors), ...
    'scaleStrideFactors must match number of scales');
assert(numel(params.scales) == numel(params.scaleGradAdjust), ...
    'scaleGradAdjust must match number of scales');

model = load_model(modelFile);
files = collect_images(imageDir);
if isempty(files)
    error('detect_pedestrians:NoImages', 'No images found in %s', imageDir);
end

rng('shuffle');
imgPath = files{randi(numel(files))};
Iorig = imread(imgPath);
scaleDown = min(1, params.maxLongSide / max(size(Iorig,1), size(Iorig,2)));
if scaleDown < 1
    Iorig = imresize(Iorig, scaleDown);
end

IprocFull = preprocess_full_image(Iorig, cfg);
[bboxes, scores] = multi_scale_detect(IprocFull, cfg, model, params);
[bboxes, scores] = filter_aspect_ratio(bboxes, scores, cfg, params.aspectTolerance);
[bboxes, scores] = non_max_suppression(bboxes, scores, params.nmsThreshold);
[bboxes, scores] = suppress_nested_boxes(bboxes, scores, params.nestedContainment);
[bboxes, scores] = limit_detections(bboxes, scores, params.maxDetectionsToShow);

gtBoxes = [];
if drawGT && exist(gtFile, 'file')
    gtBoxes = lookup_ground_truth(gtFile, imgPath);
    if scaleDown < 1 && ~isempty(gtBoxes)
        gtBoxes(:, [1 3]) = gtBoxes(:, [1 3]) * scaleDown;
        gtBoxes(:, [2 4]) = gtBoxes(:, [2 4]) * scaleDown;
    end
end

figure('Name', sprintf('Detections: %s', strip_path(imgPath)), 'NumberTitle', 'off');
imshow(Iorig);
title(sprintf('%s | %d detections', strip_path(imgPath), size(bboxes, 1)), 'Interpreter', 'none');
hold on;
for i = 1:size(bboxes, 1)
    rectangle('Position', bboxes(i, :), 'EdgeColor', 'g', 'LineWidth', 2);
    text(bboxes(i,1), bboxes(i,2)-5, sprintf('%.2f', scores(i)), 'Color', 'g', ...
        'FontSize', 8, 'FontWeight', 'bold', 'BackgroundColor', 'k');
end
if ~isempty(gtBoxes)
    for j = 1:size(gtBoxes, 1)
        rectangle('Position', gtBoxes(j, :), 'EdgeColor', 'r', 'LineWidth', 1.5, 'LineStyle', '--');
    end
end
hold off;

fprintf('Processed %s with %d detections.\n', strip_path(imgPath), size(bboxes,1));

%% ------------------------------------------------------------------------
function model = load_model(modelFile)
data = load(modelFile);
if isfield(data, 'results') && isfield(data.results, 'model')
    model = data.results.model;
elseif isfield(data, 'model')
    model = data.model;
else
    error('detect_pedestrians:MissingModel', 'Could not find model in %s', modelFile);
end
end

function files = collect_images(imageDir)
exts = {'*.png','*.jpg','*.jpeg','*.bmp'};
files = {};
for k = 1:numel(exts)
    listing = dir(fullfile(imageDir, exts{k}));
    files = [files; fullfile(imageDir, {listing.name})']; %#ok<AGROW>
end
files = files(:);
end

function Iout = preprocess_full_image(Iin, cfg)
if ndims(Iin) == 3
    Igray = rgb2gray(Iin);
else
    Igray = Iin;
end
Igray = im2double(Igray);
if isfield(cfg, 'equalizeHist') && cfg.equalizeHist
    Igray = histeq(Igray);
end
if isfield(cfg, 'denoise') && cfg.denoise
    kernel = ones(3)/9;
    Igray = filter2(kernel, Igray, 'same');
end
Iout = im2uint8(Igray);
end

function [allBboxes, allScores] = multi_scale_detect(I, cfg, model, params)
allBboxes = zeros(0,4);
allScores = zeros(0,1);
for idx = 1:numel(params.scales)
    scale = params.scales(idx);
    penalty = params.scalePenalties(idx);
    if scale <= 0
        continue;
    end
    Iscaled = imresize(I, scale);
    stride = max(4, round(params.baseStride .* params.scaleStrideFactors(idx)));
    gradThresh = max(0, params.gradientThreshold + params.scaleGradAdjust(idx));
    [b, sc] = sliding_window_detect(Iscaled, cfg, model, stride, ...
        params.scoreThreshold, gradThresh, penalty, scale, params);
    if isempty(b)
        continue;
    end
    b(:,1:2) = b(:,1:2) ./ scale;
    b(:,3:4) = b(:,3:4) ./ scale;
    allBboxes = [allBboxes; b]; %#ok<AGROW>
    allScores = [allScores; sc]; %#ok<AGROW>
end
end

function [bboxes, scores] = sliding_window_detect(I, cfg, model, stride, scoreThresh, gradThresh, penalty, scale, params)
winSize = cfg.targetSize;
winH = winSize(1);
winW = winSize(2);
[imgH, imgW] = size(I);
if imgH < winH || imgW < winW
    bboxes = zeros(0,4);
    scores = zeros(0,1);
    return;
end
rows = generate_positions(imgH, winH, stride(1));
cols = generate_positions(imgW, winW, stride(2));
bboxes = zeros(0,4);
scores = zeros(0,1);
for r = rows
    for c = cols
        patch = I(r:r+winH-1, c:c+winW-1);
        gradEnergy = mean_grad_energy(patch);
        if gradEnergy < gradThresh
            continue;
        end
        feat = extract_hog(patch, cfg);
        [label, rawScore] = predict(model, feat);
        score = positive_score(rawScore, model) - penalty;
        if label == 1
            boxWidth = winW / scale;
            boxHeight = winH / scale;
            aspect = boxWidth / max(1, boxHeight);
            if score < params.lowScoreCutoff
                if boxWidth < params.minWidthForLowScore
                    continue;
                end
                if score < params.highScoreOverride
                    if aspect < params.minAspectForLowScore || aspect > params.maxAspectForLowScore
                        continue;
                    end
                end
            end
        end

        if label == 1 && score >= scoreThresh
            bboxes(end+1, :) = [c, r, winW, winH]; %#ok<AGROW>
            scores(end+1, 1) = score; %#ok<AGROW>
        end
    end
end
end

function energy = mean_grad_energy(patch)
patchD = im2double(patch);
[Gx, Gy] = imgradientxy(patchD);
energy = (mean(abs(Gx(:))) + mean(abs(Gy(:)))) / 2;
end

function rows = generate_positions(imgDim, winDim, stride)
rows = 1:stride:max(1, imgDim - winDim + 1);
lastPos = imgDim - winDim + 1;
if rows(end) ~= lastPos
    rows(end+1) = lastPos;
end
rows = unique(rows);
end

function [bboxes, scores] = filter_aspect_ratio(bboxes, scores, cfg, tol)
if isempty(bboxes)
    return;
end
targetAspect = cfg.targetSize(2) / cfg.targetSize(1);
ratios = bboxes(:,3) ./ bboxes(:,4);
mask = ratios >= targetAspect * (1 - tol) & ratios <= targetAspect * (1 + tol);
bboxes = bboxes(mask, :);
scores = scores(mask);
end

function [pickedBboxes, pickedScores] = non_max_suppression(bboxes, scores, overlapThresh)
if isempty(bboxes)
    pickedBboxes = zeros(0,4);
    pickedScores = zeros(0,1);
    return;
end
if overlapThresh <= 0
    pickedBboxes = bboxes;
    pickedScores = scores;
    return;
end
x1 = bboxes(:,1);
y1 = bboxes(:,2);
x2 = x1 + bboxes(:,3) - 1;
y2 = y1 + bboxes(:,4) - 1;
areas = bboxes(:,3) .* bboxes(:,4);
[~, order] = sort(scores, 'descend');
keep = zeros(0,1);
while ~isempty(order)
    i = order(1);
    keep(end+1,1) = i; %#ok<AGROW>
    order(1) = [];
    if isempty(order)
        break;
    end
    xx1 = max(x1(i), x1(order));
    yy1 = max(y1(i), y1(order));
    xx2 = min(x2(i), x2(order));
    yy2 = min(y2(i), y2(order));
    w = max(0, xx2 - xx1 + 1);
    h = max(0, yy2 - yy1 + 1);
    overlap = (w .* h) ./ (areas(i) + areas(order) - w .* h);
    order = order(overlap <= overlapThresh);
end
pickedBboxes = bboxes(keep, :);
pickedScores = scores(keep);
end

function [bboxes, scores] = limit_detections(bboxes, scores, maxCount)
if maxCount <= 0 || isempty(bboxes)
    return;
end
[~, order] = sort(scores, 'descend');
order = order(1:min(maxCount, numel(order)));
bboxes = bboxes(order, :);
scores = scores(order);
end

function score = positive_score(rawScore, model)
if size(rawScore, 2) == 1
    score = rawScore;
    return;
end
if isfield(model, 'ClassNames')
    posIdx = find(model.ClassNames == 1, 1);
else
    posIdx = [];
end
if isempty(posIdx)
    posIdx = size(rawScore, 2);
end
score = rawScore(:, posIdx);
end

function name = strip_path(pathStr)
[~, base, ext] = fileparts(pathStr);
name = [base ext];
end

function boxes = lookup_ground_truth(gtFile, imgPath)
persistent gtMap
if isempty(gtMap)
    gtMap = parse_ground_truth(gtFile);
end
rel = relative_path(imgPath);
if isKey(gtMap, rel)
    boxes = gtMap(rel);
else
    boxes = [];
end
end

function gtMap = parse_ground_truth(gtFile)
gtMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
fid = fopen(gtFile, 'r');
if fid == -1
    warning('detect_pedestrians:MissingGT', 'Could not open %s for ground-truth overlay.', gtFile);
    return;
end
cleanup = onCleanup(@() fclose(fid));

header = fgetl(fid); %#ok<NASGU>
countLine = fgetl(fid); %#ok<NASGU>
while ~feof(fid)
    line = strtrim(fgetl(fid));
    if isempty(line)
        continue;
    end
    parts = strsplit(line);
    imgRel = parts{1};
    objCount = str2double(parts{2});
    nums = str2double(parts(3:end));
    boxes = zeros(objCount, 4);
    for i = 1:objCount
        idx = (i-1)*5 + 1;
        cx = nums(idx);
        cy = nums(idx+1);
        w = nums(idx+2);
        h = nums(idx+3);
        % dataset stores center coordinates; convert to top-left [x y w h]
        x = cx - w/2;
        y = cy - h/2;
        boxes(i, :) = [x y w h];
    end
    gtMap(imgRel) = boxes;
end
end

function rel = relative_path(absPath)
cwd = pwd;
if startsWith(absPath, cwd)
    rel = erase(absPath, [cwd filesep]);
else
    rel = absPath;
end
rel = strrep(rel, '\\', '/');
end

function [bboxes, scores] = suppress_nested_boxes(bboxes, scores, overlapFrac)
if isempty(bboxes)
    return;
end
[~, order] = sort(scores, 'descend');
keepMask = true(size(bboxes,1),1);
for i = 1:numel(order)
    idx = order(i);
    if ~keepMask(idx)
        continue;
    end
    for j = i+1:numel(order)
        idy = order(j);
        if ~keepMask(idy)
            continue;
        end
        if is_contained(bboxes(idy,:), bboxes(idx,:), overlapFrac)
            keepMask(idy) = false;
        elseif is_contained(bboxes(idx,:), bboxes(idy,:), overlapFrac)
            keepMask(idx) = false;
            break;
        end
    end
end
bboxes = bboxes(keepMask, :);
scores = scores(keepMask);
end

function flag = is_contained(inner, outer, overlapFrac)
% Returns true if inner lies mostly inside outer.
innerArea = inner(3) * inner(4);
outerArea = outer(3) * outer(4);
if innerArea == 0 || outerArea == 0
    flag = false;
    return;
end
% intersection
xx1 = max(inner(1), outer(1));
yy1 = max(inner(2), outer(2));
xx2 = min(inner(1)+inner(3), outer(1)+outer(3));
yy2 = min(inner(2)+inner(4), outer(2)+outer(4));
w = max(0, xx2 - xx1);
h = max(0, yy2 - yy1);
interArea = w * h;

% require most of inner covered by outer and inner smaller
flag = (interArea / innerArea) >= overlapFrac && innerArea <= outerArea;
end

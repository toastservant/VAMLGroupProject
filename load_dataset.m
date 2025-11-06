function [files, labels] = load_dataset(cfg)
% load_dataset Gather positive/negative image paths with deterministic order

exts = {'*.png', '*.jpg', '*.jpeg', '*.bmp'};
posFiles = collect_files(cfg.posDir, exts);
negFiles = collect_files(cfg.negDir, exts);

if isempty(posFiles)
    warning('load_dataset:EmptyPos', 'No positive images found in %s', cfg.posDir);
end
if isempty(negFiles)
    warning('load_dataset:EmptyNeg', 'No negative images found in %s', cfg.negDir);
end

files = [posFiles; negFiles];
labels = [ones(numel(posFiles), 1); -ones(numel(negFiles), 1)];

rng(cfg.seed, 'twister'); % deterministic shuffle
perm = randperm(numel(files));
files = files(perm);
labels = labels(perm);
end

function fileList = collect_files(rootDir, exts)
fileList = {};
if ~isfolder(rootDir)
    warning('load_dataset:MissingDir', 'Directory not found: %s', rootDir);
    return;
end
for i = 1:numel(exts)
    listing = dir(fullfile(rootDir, exts{i}));
    paths = fullfile(rootDir, {listing.name});
    fileList = [fileList; paths(:)']; %#ok<AGROW>
end
fileList = fileList(:);
end

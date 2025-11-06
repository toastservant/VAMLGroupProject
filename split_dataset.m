function [trainIdx, testIdx] = split_dataset(labels, testRatio, seed)
% split_dataset Balanced train/test partition with fixed seed

% Used fixed random seed for reproducibility
rng(seed, 'twister');

trainIdx = false(size(labels));
testIdx = false(size(labels));

% Ensure balanced split of positives and negatives
pos = find(labels == 1);
neg = find(labels == -1);

posTestCount = max(1, round(numel(pos) * testRatio));
negTestCount = max(1, round(numel(neg) * testRatio));

pos = pos(randperm(numel(pos)));
neg = neg(randperm(numel(neg)));

testIdx(pos(1:posTestCount)) = true;
trainIdx(pos(posTestCount+1:end)) = true;

testIdx(neg(1:negTestCount)) = true;
trainIdx(neg(negTestCount+1:end)) = true;
end

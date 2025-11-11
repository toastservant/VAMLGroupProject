function model = svm_classifier(XTrain, YTrain, opts)
% svm_classifier Train an SVM for pedestrian classification (HOG features expected)
%   model = svm_classifier(XTrain, YTrain) trains a linear SVM with
%   standardized features.
%   model = svm_classifier(..., opts) lets you specify:
%       opts.kernel        -> KernelFunction (default 'linear')
%       opts.boxConstraint -> BoxConstraint (default 1)
%       opts.kernelScale   -> KernelScale (default 'auto')
%       opts.standardize   -> Standardize flag (default true)
%
% XTrain : N x D matrix of feature vectors (e.g., HOG descriptors)
% YTrain : N x 1 labels (+1 / -1)

if nargin < 3 || isempty(opts)
    opts = struct();
end

if ~isfield(opts, 'kernel')
    opts.kernel = 'linear';
end
if ~isfield(opts, 'boxConstraint')
    opts.boxConstraint = 1;
end
if ~isfield(opts, 'kernelScale')
    opts.kernelScale = 'auto';
end
if ~isfield(opts, 'standardize')
    opts.standardize = true;
end

fprintf('[SVM] Training classifier (kernel=%s, C=%g)...\n', opts.kernel, opts.boxConstraint);
model = fitcsvm(XTrain, YTrain, ...
    'KernelFunction', opts.kernel, ...
    'BoxConstraint', opts.boxConstraint, ...
    'KernelScale', opts.kernelScale, ...
    'Standardize', opts.standardize, ...
    'ClassNames', [-1 1]);

fprintf('[SVM] Training complete.\n');
end

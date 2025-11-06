function Iout = preprocess_image(Iin, cfg)
% preprocess_image Apply grayscale enhancement pipeline from Prac 1 & 2

% Convert to grayscale if RGB
if ndims(Iin) == 3
    Igray = rgb2gray(Iin); % convert to grayscale
else
    Igray = Iin;
end

% Work in double for filtering/equalisation
Igray = im2double(Igray);

% Resize to target detector window
Iproc = imresize(Igray, cfg.targetSize);

% Equalise histogram (Prac 1) if requested
if isfield(cfg, 'equalizeHist') && cfg.equalizeHist
    Iproc = histeq(Iproc);
end

% Apply 3x3 average filter (Prac 2) if requested
if isfield(cfg, 'denoise') && cfg.denoise
    kernel = ones(3) / 9;
    Iproc = filter2(kernel, Iproc, 'same');
end

% Return as uint8 grayscale image
Iout = im2uint8(Iproc);
end

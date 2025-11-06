function cfg = config()
% config Prepare preprocessing options from Prac 1 & 2
cfg.posDir = fullfile(pwd, 'crops', 'pos'); % positive crops
cfg.negDir = fullfile(pwd, 'crops', 'neg'); % negative crops
cfg.targetSize = [128 64]; % [rows cols] for 64x128 window
cfg.equalizeHist = true; % histeq toggle (Prac 1)
cfg.denoise = false; % 3x3 average filter (Prac 2)
cfg.seed = 42; % fixed seed for reproducibility
end

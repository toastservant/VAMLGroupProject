function f = extract_rawpixels(I)
% Convert image to normalised row vector (raw pixel features)
% TODO: Replace with HOG after Practical 5

I = im2double(I); % map uint8 to [0,1]
f = I(:)'; % flatten column-wise into 1xN
end

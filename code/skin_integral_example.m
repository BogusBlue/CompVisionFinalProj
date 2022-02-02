%%
restoredefaultpath;
clear all;
close all;

% Replace this path with your cs4379c-fall2021 git repository path in your system.
repo_path = 'C:\Users\vangelis\Files\Academia\Teaching\TxState\CS4379C-Fall2021\Git\cs4379c-fall2021';

s = filesep; % This gets the file separator character from the  system

addpath([repo_path s 'Code' s '00_common' s '00_detection'])
addpath([repo_path s 'Code' s '00_common' s '00_images'])
addpath([repo_path s 'Code' s '00_common' s '00_utilities'])
addpath([repo_path s 'Code' s '17_boosting'])
addpath([repo_path s 'Data' s '00_common_data' s 'frgc2_b'])
cd([repo_path s 'Data' s '00_common_data'])

%%
% Load skin detection histograms
negative_histogram = read_double_image('negatives.bin');
positive_histogram = read_double_image('positives.bin');

image = imread('v1b.bmp');
figure; imshow(image, []);

% Apply skin detection
skin_detection = detect_skin(image, positive_histogram,  negative_histogram);
skin_threshold = 0.6; % threshold to binarize skin detection image
skin_pixels = skin_detection > skin_threshold;
figure; imshow(skin_pixels, []);

%%
% Scale original image.
% A scale s means that the template size must be multipled by s in
% order to match the occurrence of the object in the image.
% Since we don't want to scale the template but the original image we
% scale the image by 1/scale.
scale = 0.5;
scaled_image = imresize(image, 1/double(scale), 'bilinear');
scaled_skin_pixels = imresize(skin_pixels, 1/double(scale), 'bilinear');

% Calculate skin integral image on scaled skin detection image
scaled_skin_integral = integral_image(scaled_skin_pixels);

% Let's count skin pixels using scaled skin detection integral image on a
% particular window

%%
% Specify window offset (top-left corner)
top_offset = 700; 
left_offset = 1050;

% Specify window size 
wrows = 350;
wcols = 300;

%%
% Counting the skin pixels without the use of integral image
tic;
window = scaled_skin_pixels(top_offset:top_offset+wrows-1, left_offset:left_offset+wcols-1);
sum_skin_pixels = sum(sum(window))
toc;

%%
% Counting the skin pixels with the use of integral image
tic;
area1 = scaled_skin_integral(top_offset - 1, left_offset - 1);
area2 = scaled_skin_integral(top_offset + wrows - 1, left_offset + wcols - 1);
area3 = scaled_skin_integral(top_offset + wrows - 1, left_offset - 1);
area4 = scaled_skin_integral(top_offset - 1, left_offset + wcols - 1);

result = area1 + area2 - area3 - area4
toc;

% Note: counting skin pixels using the integral skin image should be faster
% than counting the skin pixels on the original skin detection image,
% especially for large window sizes. However, times measured in practice
% may vary due to a number of factors.

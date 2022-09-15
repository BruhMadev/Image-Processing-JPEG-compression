close all
clear all

% reading the raw image file
img = imread('raw_image.jpg');
%figure, imshow(img);

% Converting the image to grayscale
img_gray = rgb2gray(img);
imwrite(img_gray, 'original_image.jpg');
figure('name', 'Original image in grayscale'), imshow(img_gray);
[row, column] = size(img_gray);

% inserting different types of noises
% gaussian white noise
imgn1 = imnoise(img_gray, 'gaussian');
figure('name', 'Image with Gaussian noise'), imshow(imgn1);

% salt pepper noise
imgn2 = imnoise(img_gray,'salt & pepper');
figure('name', 'Image with Salt and Pepper noise'), imshow(imgn2);

% speckle noise
imgn3 = imnoise(img_gray,'speckle');
figure('name', 'Image with Speckle noise'), imshow(imgn3);


% removing salt and pepper noise using median filter
% taking 3x3 blocks and finding the median and replacing 
% the center value with the median value
imgn2_recovered = imgn2;
for i = 0 : row-1
    for j = 0 : column-1
        i_min = max(1, i);
        j_min = max(1, j);
        i_max = min(i+2, row); 
        j_max = min(j+2, column);
        median_block = imgn2(i_min:i_max, j_min:j_max);
        median_vector = median_block(:);
        median_value = median(median_vector); 
        imgn2_recovered(i+1, j+1) = median_value;
    end
end

% image after being recovered using median filter from SNP noise
figure('name', 'Recovered image from SNP noise using median filter'), imshow(imgn2_recovered);

% Removing Gaussian noise using wiener filter
imgn1_recovered = wiener2(imgn1, [5 5]);
figure('name', 'Recovered image from Gaussian noise using wiener filter'), imshow(imgn1_recovered);

% Removing Speckle noise using gaussian filter
% Gaussian filter uses a kernel that has the probability distribution in
% the form of a gaussian curve, the kernel is used to calculate weighted
% averages throughout the image

sigma = 1; % this standard deviation decides the width of the bell curve
cut_off = 3*sigma;
gaussian_filter = fspecial('gaussian', 2*cut_off+1, sigma); % here 2*cutoff+1 is the kernel size
imgn3_recovered = conv2(imgn3, gaussian_filter, 'same')/256;
figure('name', 'Recovered image from Speckle noise using gaussian filter'), imshow(imgn3_recovered);


% compressing the image using DCT (jpeg compression)
F = dct2(img_gray);
figure, imshow(F*0.01);
ff = idct2(F);
figure, imshow(ff/255);


DF = zeros(row, column);
DFF = DF;
IDF = DF;
IDFF = DF;
depth = 3;
N = 8;

for i = 1 : N : row
    for j = 1: N : column
        f = img_gray(i:i+N-1, j:j+N-1);
        df = dct2(f);
        DF(i:i+N-1, j:j+N-1) = df;
        dff = idct2(df);
        DFF(i:i+N-1, j:j+N-1) = dff;
        
        df(N:-1:depth+1,:) = 0;
        df(:, N:-1:depth+1);
        IDF(i:i+N-1, j:j+N-1) = df;
        dff = idct2(df);
        IDFF(i:i+N-1, j:j+N-1) = dff;
    end
end

figure('name', 'compressed transform'), imshow(IDF/255);
figure('name', 'rebuilt image'), imshow(IDFF/255);
A = IDFF/255;
imwrite(A, 'new_image.jpg');

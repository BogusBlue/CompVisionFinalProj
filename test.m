clear all;
close all;

directories
addpath(code_directory)
addpath(data_directory)
addpath(training_directory)

load class_results.mat;
load weak_class_results.mat;

% identify testing faces
face_images = dir(append(training_directory, '/test_cropped_faces/*.bmp'));      
face_file_count = length(face_images);

% need the labels to identify face/not face
% won't work without the extra ", 1)" idk why
labels = zeros(face_file_count * 2, 1);
labels(1:face_file_count,:) = 1;
labels(face_file_count + 1:end,:) = -1;
%labels((face_file_count * 2) + 1:end,:) = 1;
% make the 3D testing data array
test_data = zeros(100, 100, face_file_count * 2);

% read files into 3D testing data array
for i=1:face_file_count
    filename = face_images(i).name;
    train_image = read_gray(append(training_directory, '/test_cropped_faces/', filename));
    test_data(:, :, i) = train_image;
end

% now find testing nonfaces
nonface_images = dir(append(training_directory, '/test_nonfaces/*.JPG'));      
nonface_file_count = length(nonface_images);
windows_per_nonface = ceil(face_file_count / nonface_file_count);

% iterate through and find the nonface images to train with
ex_counter = face_file_count + 1;
for i=1:nonface_file_count
    filename = nonface_images(i).name;
    start_image = read_gray(append(training_directory, '/test_nonfaces/', filename));
    % find windows of size = to the face images, for better comparisons
    for j=1:windows_per_nonface
        if ex_counter > face_file_count * 2
            break;
        end
        train_image = start_image(1+j:100+j, 1+j:100+j);
        test_data(:, :, ex_counter) = train_image;
        ex_counter = ex_counter + 1;
    end
end

%{
This section is from an attempt to test with all test files. We ran into an
issue where, in order to test the images properly we have to take windows
of the image of size 100,100, but not every subwindow has a face, so it's
impossible to test this way and it ruins the numbers we were getting. In
order to test properly we would have to know specifically which windows
have faces, but with the sheer number of windows, this is practically
impossible.

% now find testing face images
test_face_images = dir(append(training_directory, '/test_face_photos/*.JPG'));      
test_face_file_count = length(test_face_images);
windows_per_test_face = ceil(face_file_count / test_face_file_count);

% iterate through and find the nonface images to train with
ex_counter = (face_file_count * 2) + 1;
for i=1:test_face_file_count
    filename = test_face_images(i).name;
    start_image = read_gray(append(training_directory, '/test_face_photos/', filename));
    % find windows of size = to the face images, for better comparisons
    for j=1:windows_per_test_face
        if ex_counter > face_file_count * 3
            break;
        end
        train_image = start_image(1+j:100+j, 1+j:100+j);
        test_data(:, :, ex_counter) = train_image;
        ex_counter = ex_counter + 1;
    end
end
%}


% testing base adaboosted and bootstrapped results from train.m
std_counter = 0;
false_positive_count = 0;
true_positive_count = 0;
false_negative_count = 0;
true_negative_count = 0;

true_class = zeros(face_file_count * 2);
predicted_class = zeros(face_file_count * 2);

% for all the images in the testing set
for i=1 : face_file_count * 2
    % load up the test image and run the test to find match
    test_image = test_data(:, :, i);
    test_result = apply_classifier_aux(test_image, final_classifier, weak_classifiers, [100,100]);
    
    predicted_class(i) = labels(i, 1);
    
    % evaluate test results
    minimum = min(test_result, [], 'all');
    maximum = max(test_result, [], 'all');
    if minimum < 0 && labels(i, 1) == -1
        true_negative_count = true_negative_count + 1;
        true_class(i) = -1;
    elseif maximum > 0 && labels(i, 1) == 1
        true_positive_count = true_positive_count + 1;
        true_class(i) = 1;
    elseif maximum > 0 && labels(i, 1) == -1
        false_positive_count = false_positive_count + 1;
        true_class(i) = 1;
    elseif minimum < 0 && labels(i, 1) == 1
        false_negative_count = false_negative_count + 1;
        true_class(i) = -1;
    end
    std_counter = std_counter + 1;
end

% if you don't do *2 then eveything is 50% when displaying results
true_positive_rate = true_positive_count / std_counter*2;
true_negative_rate = true_negative_count / std_counter*2;
false_positive_rate = false_positive_count / std_counter*2;
false_negative_rate = false_negative_count / std_counter*2;

% print results
disp('Standard Results')
msg = ['Total Number of Tests: ', num2str(face_file_count * 2)];
disp(msg);
msg = ['False Positive Rate: ', num2str(false_positive_rate*100), '%'];
disp(msg);
msg = ['False Negative Rate: ', num2str(false_negative_rate*100), '%'];
disp(msg);
msg = ['Positive Success Rate: ', num2str(true_positive_rate*100), '%'];
disp(msg);
msg = ['Negative Success Rate: ', num2str(true_negative_rate*100), '%'];
disp(msg);

% I tried to get a confusion matrix working, but it kept giving me errors
% such as 'First two arguments must be vectors or character matrices.' I
% also ran out of time

%confusion_matrix = confusionmat(true_class, predicted_class);
%confusion_matrix


% this part was to show examples of when the program struggles and succeeds while
% only using normal AdaBoost (without bootstrapping)
gray_frame = read_gray(append(training_directory, '/test_face_photos/IMG_3793.JPG'));
drawing_picture = apply_classifier_aux(gray_frame, final_classifier, weak_classifiers, [100,100]);
center_value = max(drawing_picture, [], 'all');
[center_row, center_col] = find(drawing_picture == center_value);
rectangle_picture = draw_rectangle2(gray_frame, center_row, center_col, 100, 100);
figure(1);
imshow(rectangle_picture-125, []);

gray_frame = read_gray(append(training_directory, '/test_face_photos/the-lord-of-the-rings_poster.JPG'));
drawing_picture = apply_classifier_aux(gray_frame, final_classifier, weak_classifiers, [100,100]);
center_value = max(drawing_picture, [], 'all');
[center_row, center_col] = find(drawing_picture == center_value);
rectangle_picture = draw_rectangle2(gray_frame, center_row, center_col, 100, 100);
figure(2);
imshow(rectangle_picture-125, []);            




% skin detection
negative_histogram = read_double_image('negatives.bin');
positive_histogram = read_double_image('positives.bin');

% comparing single skin detection time with regular detection
color_frame = double(imread(append(training_directory, '/test_face_photos/clintonAD2505_468x448.JPG')));
gray_frame = read_gray(append(training_directory, '/test_face_photos/clintonAD2505_468x448.JPG'));
gray_frame_copy = read_gray(append(training_directory, '/test_face_photos/clintonAD2505_468x448.JPG'));
[x, y] = size(gray_frame);
tic;
skin = detect_skin(color_frame, positive_histogram,  negative_histogram);
skin = skin > .25;
%skin_frame = skin & gray_frame;
for i=1 : x
    for j=1 : y
        if(~skin(i,j))
            gray_frame(i,j) = 0;
        end
    end
end

accuracy_counter = 0;
std_counter = 0;


test_result = apply_classifier_aux(gray_frame, final_classifier, weak_classifiers, [100,100]);
elapsedTime = toc;
regular_time = elapsedTime;
std_counter = std_counter + 1;
maximum = max(test_result, [], 'all');
if minimum < 0
    % Do nothing
else
    accuracy_counter = accuracy_counter + 1;
end
tic;
test_result = apply_classifier_aux(gray_frame_copy, final_classifier, weak_classifiers, [100,100]);
elapsedTime = toc;
skin_time = elapsedTime;


% I did try to get this part to look like the part above, but I ran into
% memory issues with making an array large enough to store large amounts of
% rgb images, so this is all I could do for this section

% However, I was able to get the following code to work with the
% nonface_filenames.m file so that only one image is loaded in at a time,
% so tests can be run

skin_test_filenames = nonface_filenames();
[skin_test_filename_count, ~] = size(skin_test_filenames);
for i=1 : skin_test_filename_count
    color_frame = double(imread(append(training_directory, '/test_nonfaces/', skin_test_filenames{i})));
    gray_frame = read_gray(append(training_directory, '/test_nonfaces/', skin_test_filenames{i}));
    skin = detect_skin(color_frame, positive_histogram,  negative_histogram);
    skin = skin > .25; 
    skin_frame = skin & gray_frame;
    
    % While this test isn't exactly a 1:1 comparison with the other tests,
    % making a large array of rgb test images in matlab is not possible, as
    % I have gotten many array/memory size errors while trying to do that,
    % so this is the next best thing
    test_result = apply_classifier_aux(skin_frame, final_classifier, weak_classifiers, [100,100]);
    
    % evaluate test results
    minimum = min(test_result, [], 'all');
    if minimum < 0
        accuracy_counter = accuracy_counter + 1;
    else
        % Do nothing
    end
    std_counter = std_counter + 1;
end
% a caviet for the skin detection time is that by the nature of the code,
% and not having the images preloaded, it will always take longer
% regardless

skin_accuracy_rate = accuracy_counter / std_counter;

% print results
disp(' ')
disp('Skin Detection Results')
msg = ['Total Number of Tests: ', num2str(skin_test_filename_count+2)];
disp(msg);
msg = ['Accuracy Rate: ', num2str(skin_accuracy_rate*100), '%'];
disp(msg);
msg = ['Skin detection compute time was ', num2str(regular_time - skin_time), ' seconds faster than base compute time'];
disp(msg);

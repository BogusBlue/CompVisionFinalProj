clear all;
close all;
restoredefaultpath;

directories
addpath(code_directory)
addpath(data_directory)
addpath(training_directory)

% building the training set
% identify training faces
face_images = dir(append(training_directory, '/training_faces/*.bmp'));      
face_file_count = length(face_images);

% need the labels to identify face/not face
% won't work without the extra ", 1)" idk why
labels = zeros(face_file_count * 2, 1);
labels(1:face_file_count,:) = 1;
labels(face_file_count + 1:end,:) = -1;
% make the 3D training data array
training_data = zeros(100, 100, face_file_count * 2);

% read files into 3D training data array
for i=1:face_file_count
    filename = face_images(i).name;
    train_image = read_gray(append(training_directory, '/training_faces/', filename));
    training_data(:, :, i) = train_image;
end

% now find training nonfaces
%nonface_images = nonface_filenames;
nonface_images = dir(append(training_directory, '/training_nonfaces/*.JPG'));      
nonface_file_count = length(nonface_images);
windows_per_nonface = ceil(face_file_count / nonface_file_count);

% iterate through and find the nonface images to train with
ex_counter = face_file_count + 1;
for i=1:nonface_file_count
    filename = nonface_images(i).name;
    start_image = read_gray(append(training_directory, '/training_nonfaces/', filename));
    % find windows of size = to the face images, for better comparisons
    for j=1:windows_per_nonface
        if ex_counter > face_file_count * 2
            break;
        end
        train_image = start_image(1+j:100+j, 1+j:100+j);
        training_data(:, :, ex_counter) = train_image;
        ex_counter = ex_counter + 1;
    end
end

% making random weak classifiers (from main_script 250-254)
ex_number = 1000; 
weak_classifiers = cell(1, ex_number);
for i = 1:ex_number
    % make them 100 by 100, the size of the cropped faces for simplicity
    weak_classifiers{i} = generate_classifier(100, 100);
end

% need integrals of training data for evaluation
ex_number2 = 2 * face_file_count;
integrals = zeros(100, 100, ex_number2);
for i=1:ex_number2
    integrals(:, :, i) = integral_image(training_data(:, :, i));
end

% numel = ex_number of elements in an array (broke if use 2000)
num_classifiers = numel(weak_classifiers);
class_eval_results = zeros(num_classifiers, ex_number2);

for example = 1:ex_number2
    % integral = integrals of training data
    integral = integrals(:, :, example);
    for feature = 1:num_classifiers
        classifier = weak_classifiers {feature};
        % evaluate all of the classifiers in order to choose which to use
        class_eval_results(feature, example) = eval_weak_classifier(classifier, integral);
    end
end

% first AdaBoost training session
first_results = [class_eval_results(:, 1:300) class_eval_results(:, face_file_count+1:face_file_count+300)];
first_labels = zeros(600, 1);
first_labels(1:300, 1) = 1;
first_labels(301:600, 1) = -1;
classifier = AdaBoost(first_results, first_labels, 9);
%final_classifier = classifier;

% identifying missclassified items for bootstrapping
disqualified_images = zeros(2 * face_file_count, 1);
% we'll just keep a random set of images
disqualified_images(1:300, 1) = 1;
disqualified_images(face_file_count+1:face_file_count+300, 1) = 1;

% deterimining what images to take out of the training set to make the
% training set more difficult
num_disqualified = 0;
for i=1:face_file_count * 2
    if disqualified_images(i, 1) == 0
        % the following line returns a matrix of values that match what
        % we're looking for (classifiers)
        class = apply_classifier_aux(training_data(:, :, i), classifier, weak_classifiers, [100, 100]);
        % find the max and min of the returned matrix
        minimum = min(class, [], 'all');
        maximum = max(class, [], 'all');
        
        % if the minimum value is negative and the label is negative, or if
        % the maximum value is positive and the label is positive, it
        % guessed it right so it should be disqualified from the next
        % training set
        if minimum < 0 && labels(i, 1) == -1
            disqualified_images(i, 1) = 1;
            num_disqualified = num_disqualified + 1;
        elseif maximum > 0 && labels(i, 1) == 1
            disqualified_images(i, 1) = 1;
            num_disqualified = num_disqualified + 1;
        end
    end
end

% adding training data into the training set
face_quota = 300; % doesn't necessarily have to be 300
nonface_quota = 300;
new_train_results = first_results;
new_train_labels = first_labels;
for i=1:face_file_count * 2
    % for every non-initializaed disqualified image
    if disqualified_images(i, 1) == 0
        % if is a nonface and the nonface quota hasn't been met
        if labels(i, 1) == -1 && nonface_quota > 0
            % add results and lables to the new sets
            nonface_quota = nonface_quota - 1;
            new_train_results = [new_train_results class_eval_results(:, i)];
            new_train_labels = [new_train_labels ; labels(i, 1)];
            disqualified_images(i, 1) = 1;
        
        % otherwise if is a face and the face quota hasn't been met
        elseif labels(i, 1) == 1 && face_quota > 0
            % also add results and lables to the new sets
            face_quota = face_quota - 1;
            new_train_results = [new_train_results class_eval_results(:, i)];
            new_train_labels = [new_train_labels ; labels(i, 1)];
            disqualified_images(i, 1) = 1;
        end
    end
end
% results in full new test set

% Adaboost again, this time with double the rounds now that we have some
% difficult data
mid_classifier = AdaBoost(new_train_results, new_train_labels, 18);

% final round of throwing out data
num_added = 0;
for i=1:face_file_count * 2
    % test the new results
    class = apply_classifier_aux(training_data(:, :, i), mid_classifier, weak_classifiers, [100, 100]);
    minimum = min(class, [], 'all');
    maximum = max(class, [], 'all');

    if minimum < 0 && labels(i, 1) == -1
        % Do nothing
    elseif maximum > 0 && labels(i, 1) == 1
        % Do nothing
    else
        % misclassified, so add to the new training set to run again
        new_train_results = [new_train_results class_eval_results(:, i)];
        new_train_labels = [new_train_labels ; labels(i, 1)];
    end
end

% final training session using 24 rounds on the most difficult data in the
% training set(s) (had some issues with adding more data)
final_classifier = AdaBoost(new_train_results, new_train_labels, 24);



% Save the stuff
save class_results final_classifier;
save weak_class_results weak_classifiers;
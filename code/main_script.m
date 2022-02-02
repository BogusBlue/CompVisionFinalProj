%%

% The addpath and cd lines are the only lines in the code that you may have
% to change, in order to make the rest of the code work. Adjust
% the paths to reflect the locations where you have stored the code 
% and data in your computer.

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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%  Rectangle filters
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% type 1

clear; close all;
photo = read_gray('faces5.bmp');
figure(1); imshow(photo, []);
rec_filter = rectangle_filter1(1, 1);
result = imfilter(photo, rec_filter, 'same', 'symmetric');
figure(2); imshow(result, []);

rec_filter2 = rectangle_filter1(20, 10);
result2 = imfilter(photo, rec_filter2, 'same', 'symmetric');
figure(3); imshow(result2, []);


%%
% type 2

clear; close all;
photo = read_gray('faces5.bmp');
figure(1); imshow(photo, []);
rec_filter = rectangle_filter2(1, 1);
result = imfilter(photo, rec_filter, 'same', 'symmetric');
figure(2); imshow(result, []);

rec_filter2 = rectangle_filter2(20, 10);
result2 = imfilter(photo, rec_filter2, 'same', 'symmetric');
figure(3); imshow(result2, []);

%%
% type 3

clear; close all;
photo = read_gray('faces5.bmp');
figure(1); imshow(photo, []);
rec_filter = rectangle_filter3(1, 1);
result = imfilter(photo, rec_filter, 'same', 'symmetric');
figure(2); imshow(result, []);

rec_filter2 = rectangle_filter3(10, 7);
result2 = imfilter(photo, rec_filter2, 'same', 'symmetric');
figure(3); imshow(result2, []);

%%
% type 4

clear; close all;
photo = read_gray('faces5.bmp');
figure(1); imshow(photo, []);
rec_filter = rectangle_filter4(1, 1);
result = imfilter(photo, rec_filter, 'same', 'symmetric');
figure(2); imshow(result, []);

rec_filter2 = rectangle_filter4(10, 7);
result2 = imfilter(photo, rec_filter2, 'same', 'symmetric');
figure(3); imshow(result2, []);

%%
% type 5

clear; close all;
photo = read_gray('faces5.bmp');
figure(1); imshow(photo, []);
rec_filter = rectangle_filter5(1, 1);
result = imfilter(photo, rec_filter, 'same', 'symmetric');
figure(2); imshow(result, []);

rec_filter2 = rectangle_filter5(10, 7);
result2 = imfilter(photo, rec_filter2, 'same', 'symmetric');
figure(3); imshow(result2, []);

%%

% computing the integral image of some small input

clear; close all;
a = [1 2
     3 4
     5 6];
 
b = integral_image(a)

%%
clear; close all;
A = read_gray('faces5.bmp');
figure(1); imshow(A, []);
B = integral_image(A);

F = rectangle_filter1(50, 40);
figure(2); imshow(F, []);

tic; responses = imfilter(A, F, 'same', 'symmetric'); toc
figure(3); imshow(responses, []);

%%
responses(125, 240)
sum(sum(A(101:150, 201:280) .* F))
sum(sum(A(101:150, 201:240) )) - sum(sum(A(101:150, 241:280)))

%%
tic; sum(sum(A(101:150, 201:240))); toc

tic; B(150,240)+B(100,200)-B(150,200)-B(100,240); toc

%%

clear all;
load examples1000;

%%

% testing evaluation of a random classifier of type 1.

wc = generate_classifier1(face_vertical, face_horizontal); 
disp(wc); 
disp(wc{1}); 
disp(wc{2});

% disp(wc{9});
%%%

index = 1;
face = faces(:,:,index);
integral = face_integrals(:, :, index);

response = eval_weak_classifier(wc, integral)

% verification:
top = wc{7};
left = wc{8};
bottom = top + wc{5} - 1;
right = left + 2 * wc{6} - 1;
rec_filter = wc{9};
response2 = sum(sum(face(top:bottom, left:right) .* rec_filter))


%%

% testing evaluation of a random classifier of type 2.

wc = generate_classifier2(face_vertical, face_horizontal); 
disp(wc); 
disp(wc{1}); 
disp(wc{2});

%%%

index = 1;
face = faces(:,:,index);
integral = face_integrals(:, :, index);

response = eval_weak_classifier(wc, integral)

% verification:
top = wc{7};
left = wc{8};
bottom = top + 2 * wc{5} - 1;
right = left + 1 * wc{6} - 1;
rec_filter = wc{9};
response2 = sum(sum(face(top:bottom, left:right) .* rec_filter))


%%

% testing evaluation of a random classifier of type 3.

wc = generate_classifier3(face_vertical, face_horizontal); 
disp(wc); 
disp(wc{1}); 
disp(wc{2});

%%%

index = 1;
face = faces(:,:,index);
integral = face_integrals(:, :, index);

response = eval_weak_classifier(wc, integral)

% verification:
top = wc{7};
left = wc{8};
bottom = top + 1 * wc{5} - 1;
right = left + 3 * wc{6} - 1;
rec_filter = wc{9};
response2 = sum(sum(face(top:bottom, left:right) .* rec_filter))


%%

% testing evaluation of a random classifier of type 4.

wc = generate_classifier4(face_vertical, face_horizontal); 
disp(wc); 
disp(wc{1}); 
disp(wc{2});

%%%

index = 1;
face = faces(:,:,index);
integral = face_integrals(:, :, index);

response = eval_weak_classifier(wc, integral)

% verification:
top = wc{7};
left = wc{8};
bottom = top + 3 * wc{5} - 1;
right = left + 1 * wc{6} - 1;
rec_filter = wc{9};
response2 = sum(sum(face(top:bottom, left:right) .* rec_filter))

%%

% choosing a set of random weak classifiers

number = 1000;
weak_classifiers = cell(1, number);
for i = 1:number
    weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
end


% save classifiers1000 weak_classifiers

%%

%  precompute responses of all training examples on all weak classifiers

clear all;
load examples1000;
load classifiers1000;

example_number = size(faces, 3) + size(nonfaces, 3);
labels = zeros(example_number, 1);
labels (1:size(faces, 3)) = 1;
labels((size(faces, 3)+1):example_number) = -1;
examples = zeros(face_vertical, face_horizontal, example_number);
examples (:, :, 1:size(faces, 3)) = face_integrals;
examples(:, :, (size(faces, 3)+1):example_number) = nonface_integrals;

classifier_number = numel(weak_classifiers);

responses =  zeros(classifier_number, example_number);

for example = 1:example_number
    integral = examples(:, :, example);
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
    disp(example)
end

% save training1000 responses labels classifier_number example_number

%%
% verify that the computed responses are correct

clear all;
load training1000
load classifiers1000
load examples1000

%%

% choose a classifier
a = random_number(1, classifier_number);
wc = weak_classifiers{a};

% choose a training image
b = random_number(1, example_number);
if (b <= size(faces, 3))
    integral = face_integrals(:, :, b);
else
    integral = nonface_integrals(:, :, b - size(faces,3));
end

% see the precomputed response
disp([a, b]);
disp(responses(a, b));
disp(eval_weak_classifier(wc, integral));



%%

clear all;
load training1000;
weights = ones(example_number, 1) / example_number;


%%
cl = random_number(1, 1000)
[error, thr, alpha] = weighted_error(responses, labels, weights, cl)

%%

weights = ones(example_number, 1) / example_number;
% next line takes about 8.5 seconds.
tic; [index error threshold] = find_best_classifier(responses, labels, weights); toc
disp([index error]);

%%

clear all;
load training1000;
load classifiers1000;
boosted_classifier = AdaBoost(responses, labels, 15);

% the above line produces this output (second column is 
% error rate of current strong classifier):
% 
%             1       0.0295       0.0295          875      -1.7467      -403.81
%             2       0.0295     0.090954          426       -1.151      -193.24
%             3       0.0175     0.094024          373      -1.1327      -395.24
%             4        0.014      0.12683          283      0.96462       982.08
%             5        0.007      0.15141         1000     -0.86181      -397.68
%             6       0.0105      0.15362          864      0.85323        63.99
%             7       0.0055      0.15379          517       0.8526          159
%             8        0.004      0.18631          654     -0.73709      -148.65
%             9       0.0035       0.1776          242      0.76634       -72.45
%            10        0.001      0.17712          559      0.76798       255.69
%            11       0.0015      0.19225          369      0.71771       67.521
%            12       0.0005      0.20436          898      0.67962         1023
%            13            0      0.16552          542     -0.80886      -256.08
%            14            0      0.16625          686      0.80622       46.174
%            15            0      0.15135          780      0.86202       346.11

%%

save boosted15 boosted_classifier

%%

load faces1000;
load nonfaces1000;

% Let's classify a couple of our face and non-face training examples. 
% A positive prediction value means the classifier predicts the input image to be
% a face. A negative prediction value means the classifier thinks it's not
% a face. Values farther away from zero means the classifier is more
% confident about its prediction, either positive or negative.

prediction = boosted_predict(faces(:, :, 200), boosted_classifier, weak_classifiers, 15)

prediction = boosted_predict(nonfaces(:, :, 500), boosted_classifier, weak_classifiers, 15)

%%

clear all;
load classifiers1000;
load boosted15; 

% load a photograph
photo = read_gray('faces4.bmp');

% rotate the photograph to make faces more upright (we 
% are cheating a bit, to save time compared to searching
% over multiple rotations).
photo2 = imrotate(photo, -10, 'bilinear');
photo2 = imresize(photo2, 0.34, 'bilinear');
figure(1); imshow(photo2, []);

% w1 and w2 are the locations of the faces, according to me.
% Used just for bookkeeping.
%w1 = photo2(40:87, 75:113);
%w2 = photo2(100:130, 47:71);

%%

tic; result = boosted_multiscale_search(photo2, 1, boosted_classifier, weak_classifiers, [31, 25]); toc
figure(2); imshow(result, []);
figure(3); imshow(max((result > 4) * 255, photo2 * 0.5), [])

%%

tic; [result, boxes] = boosted_detector_demo(photo2, 1, boosted_classifier, weak_classifiers, [31, 25], 2); toc
figure(2); imshow(result, []);

%%

% load a photograph
photo = read_gray('faces4.bmp');

% rotate the photograph to make faces more upright (we 
% are cheating a bit, to save time compared to searching
% over multiple rotations).
photo2 = imrotate(photo, -10, 'bilinear');

% w1 and w2 are the locations of the faces, according to me.
% Used just for bookkeeping.
w1 = photo2(132:218, 221:290);
w2 = photo2(299:372, 133:192);

% apply the boosted detector, and get the 
% top 10 matches. Takes a few seconds on my desktop.
[result, boxes] = boosted_detector_demo(photo2, 1:0.5:3, boosted_classifier, weak_classifiers, [31, 25], 2);
figure(1); imshow(photo2, []);
figure(2); imshow(result, [])
% correct results are: w2 at rank 25, w1 at rank 38

%%

photo = read_gray('faces5.bmp');

% w1 is the location of the face, according to me.
% Used just for bookkeeping.
w1 = photo(110:148, 100:130);

% apply the boosted detector, and get the 
% top match.
[result, boxes] = boosted_detector_demo(photo, 1:0.5:3, boosted_classifier, weak_classifiers, [31, 25], 1);
figure(1); imshow(photo, []);
figure(2); imshow(result, [])
% rank of correct result using normalized correlation is 1.

%%

photo = read_gray('vjm.bmp');

% apply the boosted detector, and get the 
% top 3 matches.
tic; [result, boxes] = boosted_detector_demo(photo, 1:0.5:3., boosted_classifier, weak_classifiers, [31, 25], 3); toc
figure(1); imshow(photo, []);
figure(2); imshow(result, [])
% rank of correct result using normalized correlation is 1.

%%

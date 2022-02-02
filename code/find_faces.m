function result = find_faces(image, scales, result_number, eigenface_number)

% function result = find_faces(image, scales, result_number, eigenface_number)
%
% first, find_faces identifies, for every position in the image, the 
% scale that maximizes the normalized correlation with the
% average face. Then, the top results are re-ranked based on how well they
% match the top eigenfaces.

load face_filter;
load final_eigens;

% start filter step
preliminary_number = 50;
preliminary_results = find_template(image, face_filter, ...
                                    scales, 0, preliminary_number);
                                
% start refine step
for number = 1:preliminary_number
    top = preliminary_results(number, 1);
    bottom = preliminary_results(number, 2);
    left = preliminary_results(number, 3);
    right = preliminary_results(number, 4);
    window = image(top:bottom, left:right);
    preliminary_results(number, 6) = ...
        pca_score(window, mean_face, eigenvectors, eigenface_number);
end

[values, indices] = sort(preliminary_results(:, 6), 'ascend');
top_indices = indices(1: result_number);
result = preliminary_results(top_indices, :);


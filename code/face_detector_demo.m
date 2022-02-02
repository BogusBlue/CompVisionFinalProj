function [result, boxes] =  face_detector_demo(image, scales, result_number, eigenface_number)

% function [result, boxes] =  face_detector_demo(image, scales, result_number, eigenface_number)

boxes = find_faces(image, scales, result_number, eigenface_number);
result = image;

for number = 1:result_number
    result = draw_rectangle1(result, boxes(number, 1), boxes(number, 2), ...
                             boxes(number, 3), boxes(number, 4));
end

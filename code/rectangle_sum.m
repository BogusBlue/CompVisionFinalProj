function result = rectangle_sum(integral, rectangle, row, col)

% function result = rectangle_sum(integral, rectangle)
%
% integral:  integral image of some image A
% rectangle: [top bottom left right]
% row: an offset that is used to adjust top and bottom
% col: an offset that is used to adjust left and right.


top = max(rectangle(1), 2); % Make sure we stay within the boundaries of the image
bottom = min(rectangle(2), size(integral, 1));
left = max(rectangle(3), 2);
right = min(rectangle(4), size(integral, 2));

area1 = integral(top+row - 2, left + col - 2);
area2 = integral(bottom + row - 1, right + col - 1);
area3 = integral(bottom + row - 1, left + col - 2);
area4 = integral(top + row - 2, right + col - 1);

result = area1 + area2 - area3 - area4;

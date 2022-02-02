function plot_points(points, figure_number, style, markersize, resize)

% function plot_points(points, figure_number)
%
% function plot_points(points, figure_number, style)
%
% function plot_points(points, figure_number, style, markersize)
%
% function plot_points(points, figure_number, style, markersize, resize)

if nargin <= 2
    style = '.';
end

if nargin <= 3
    markersize = 6;
end

if nargin <= 4
    resize = 1;
end

figure(figure_number);
plot(points(1,:), points(2, :), style, 'MarkerSize', markersize);
max_xy = max(abs(points(:)))* 1.2;
hold on;
if (resize)
    set(gca, 'XLim', [-max_xy, max_xy]);
    set(gca, 'YLim', [-max_xy, max_xy]);
    set(gca, 'XGrid', 'on');
    set(gca, 'YGrid', 'on');
    set(gca, 'PlotBoxAspectRatio', [1 1 1]);
end









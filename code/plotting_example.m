a = [0 13 22 31 43 54 64 74 85 91 100];
b = [0  9 10 13 16 19 25 30 38 59 100];

close all;
figure(1);
plot(a, b, 'k-');
hold on;
plot(a, b/2, 'r--');
plot(a, b/4, 'b:');
set(gca, 'XLim', [0 105]);
set(gca, 'YLim', [0 105]);
set(gca, 'XGrid', 'on');
set(gca, 'YGrid', 'on');
xticks([0:10:100])
yticks([0:10:100])
set(gca, 'PlotBoxAspectRatio', [1 1 1]);
legend('method1', 'method2', 'method3', 'Location', 'NorthWest');

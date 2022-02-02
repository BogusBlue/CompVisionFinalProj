function result = gaussian_probability_fast(m, sigma, values)

% function gaussian_probability_fast(m, sigma, values)
%
% faster version of gaussian_probability (if the range of values is 
% not too large), rounds all values before computing their probability

total_size = prod(size(values));
result = zeros(size(values));
sigma2squared = 2 * sigma * sigma;

values = round(values);
min_value = min(values(:)) - 1;
max_value = max(values(:));
range = max_value - min_value + 1;
counters = zeros(range, 3);
counters(:, 1) = [min_value:max_value]';

for value = (min_value+1):max_value
    index = value - min_value;
    prob = exp(-(value - m)^2 / sigma2squared);
    counters(index, 2) = prob;
end

counters(:, 2) = counters(:, 2) / (sigma * sqrt(2 * pi));

for pixel = 1:total_size
    value = values(pixel);
    index = value - min_value;
    result(pixel) = counters(index, 2);
end


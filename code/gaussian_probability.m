function result = gaussian_probability(m, sigma, values)

% function gaussian_probability(m, sigma, values)

total_size = prod(size(values));
result = zeros(size(values));
sigma2squared = 2 * sigma * sigma;

for index = 1:total_size
    value = values(index);
    result(index) = exp(-(value - m)^2 / sigma2squared);
end

result = result / (sigma * sqrt(2 * pi));

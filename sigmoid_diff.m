function g = sigmoid_diff(z)
g = exp(-z) ./ ((1.0 + exp(-z)).^2);
end

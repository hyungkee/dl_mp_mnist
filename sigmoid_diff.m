function g = sigmoid_diff(z, beta)
g = beta * exp(-beta * z) ./ ((1.0 + exp(-beta * z)).^2);
end

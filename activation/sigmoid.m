function g = sigmoid(z, beta)
g = 1.0 ./ (1.0 + exp(-beta*z));
end

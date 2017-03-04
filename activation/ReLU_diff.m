function g = ReLU_diff(z, beta)
    g = beta * (sign(z)+1)/2;
end

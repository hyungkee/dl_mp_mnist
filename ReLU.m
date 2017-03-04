function g = ReLU(z, beta) % beta는 z > 0 영역에서의 기울기
    g = beta * (z + abs(z))/2;
end

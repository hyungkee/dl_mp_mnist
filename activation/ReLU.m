function g = ReLU(z, beta) % beta�� z > 0 ���������� ����
    g = beta * (z + abs(z))/2;
end

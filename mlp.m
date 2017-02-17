function [model, mse] = mlp(X, Y, h)

h = [size(X,1);h(:);size(Y,1)];
L = numel(h);
G = cell(L-1);

epsilon_init = 0.4;
for l = 1:L-1
    G{l} = rand(h(l),h(l+1)) * epsilon_init * 2 - epsilon_init;
end

V = cell(L);
V{1} = X;

dGp = 1000/size(X,2);
dGm = 1000/size(X,2);

maxiter = 50;
mse = zeros(1,maxiter);


for iter = 1:maxiter
    fprintf('%d iterations..', iter)
    
%     forward
    for l = 2:L
        V{l} = sigmoid(G{l-1}'*V{l-1});
    end
    
%     backward
    E = V{L}-Y;
    mse(iter) = mean(dot(E(:),E(:)));

    GxD = V{L}-Y; % #(L) x m
    for l = L-1:-1:1
        
        df = V{l+1}.*(1-V{l+1}); % #(l+1) x m
        D = df.*(GxD); % #(l+1) x m
        
        VxD = V{l}*D'; % {#(l) x m} * {m x #(l+1)} = #(l) x #(l+1)
        dG = VxD; % #(l) x #(l+1)
        % 이 경우는 m개의 set들의 dG를 모두 합한 다음에 이진화를 한 것이므로 사실상 근사를 거의 안했다고 보아야 한다.
        % 이진화를 V{l}*D'과정에서 진행하는 경우부터가 문제이다.
        dG(VxD >0) = -dGm;
        dG(VxD==0) = 0;
        dG(VxD <0) = dGp;
        G{l} = G{l} + dG;
        
        GxD = G{l}*D; % #(l) x m
    end
    
    fprintf('mse : %d\n', mse(iter))
end
mse = mse(1:iter);
model.W = G;
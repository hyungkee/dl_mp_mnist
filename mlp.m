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

dGp = 0.0003/size(X,2);
dGm = 0.0003/size(X,2);

maxiter = 100;
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
                
        qD = D; % #{l+1} x m. quantized D
        qD(D >0) = -dGm;
        qD(D==0) = 0;
        qD(D <0) = dGp;
        
        dG = repmat(sum(qD,2),1,size(V{l},1))'; % {#(l+1) x #(l)}' = #(l) x #(l+1)
        G{l} = G{l} + dG;
        
        GxD = G{l}*D; % #(l) x m
    end
    
    fprintf('mse : %d\n', mse(iter))
end
mse = mse(1:iter);
model.W = G;
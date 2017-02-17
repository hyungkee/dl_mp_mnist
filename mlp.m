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

eta = 2/size(X,2);
% dGp = 0.001/size(X,2);
% dGm = 0.001/size(X,2);

    
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

    GxD = V{L}-Y;
    for l = L-1:-1:1
        
        df = V{l+1}.*(1-V{l+1});
        D = df.*(GxD);
        
        dG = V{l}*D';
        G{l} = G{l} - eta*dG;
        
        GxD = G{l}*D;
    end
    
    fprintf('mse : %d\n', mse(iter))
end
mse = mse(1:iter);
model.W = G;
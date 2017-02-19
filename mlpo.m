function [model, mse] = mlpo(X, Y, h)

% 이 함수는 online learning을 위한 함수이다.

h = [size(X,1);h(:);size(Y,1)];
L = numel(h);
G = cell(L-1);

epsilon_init = 0.4;
for l = 1:L-1
    G{l} = rand(h(l),h(l+1)) * epsilon_init * 2 - epsilon_init;
end

oV = cell(L);
V = cell(L);
V{1} = X;


m = size(X,2);

eta = 2/m;
dGp = 0.01;
dGm = 0.01;

maxiter = 50;
mse = zeros(1,maxiter);


for iter = 1:maxiter
    fprintf('[%d iterations..]', iter)
    
%  for each data
    for n = 1:m

        oV{1} = X(:,n); % #(1) x 1
%     forward
        for l = 2:L
            oV{l} = sigmoid(G{l-1}'*oV{l-1}); % #(l) x 1
        end
    
%     backward
        
        GxD = oV{L}-Y(:,n); % #(L) x 1
        for l = L-1:-1:1
            df = oV{l+1}.*(1-oV{l+1}); % #(l+1) x 1
            D = df.*(GxD); % #(l+1) x 1
            
            dG = oV{l}*D'; % #(l) x #(l+1);
            dG(dG>0) = dGm; % 이 크기만큼 결국 빼지기 때문에 dGm이다. (-eta*dG)
            dG(dG<0) = -dGp; % 이 크기만큼 결국 더해지기 떄문에 dGp이다. (-eta*dG)
            
            G{l} = G{l} - eta*dG; % #(l) x #(l+1)
            
            GxD = G{l}*D; % #(l) x 1
        end
    end
    
    
% calculate mse
    for l = 2:L
        V{l} = sigmoid(G{l-1}'*V{l-1}); % #(l) x m
    end
    E = V{L}-Y;
    mse(iter) = mean(dot(E(:),E(:)));

    fprintf('(iter : %d)mse : %d\n', iter, mse(iter))
    
end
mse = mse(1:iter);
model.W = G;
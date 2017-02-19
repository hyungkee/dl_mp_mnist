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
dGp = 0.001;
dGm = 0.001;

maxiter = 100;
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
            
            oVxD = oV{l}*D'; % #(l) x #(l+1)
            dG = zeros(size(oVxD)); % #(l) x #(l+1)
            dG(oVxD > dGm) = dGm; % 조건을 D>0으로 두게되면 oV 크기에 따른 차등이 사라져 자유도가 급감한다. 따라서 oVxD > dGm으로 둔다.
            dG(oVxD < -dGp) = -dGp; % D<0또한 위와 같다. 따라서 OVxD < -dGp로 둔다.
            
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
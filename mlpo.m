function [model, mse] = mlpo(X, Y, h, options)
% 이 함수는 online learning을 위한 함수이다.

fprintf('\n\n');
fprintf('[online learning]\n');

% option읽기
if strcmp(options.active,'sigmoid')
    active = @sigmoid;
    active_diff = @sigmoid_diff;
    fprintf(' - active function : sigmoid\n');
elseif strcmp(options.active,'ReLU')
    active = @ReLU;
    active_diff = @ReLU_diff;
    fprintf(' - active function : ReLU\n');
elseif strcmp(options.active,'linear')
    active = @linear;
    active_diff = @linear_diff;
    fprintf(' - active function : linear\n');
end
epsilon_init = options.epsilon_init;
dGp = options.dG;
dGm = options.dG;
maxiter = options.MaxIter;
eta = options.eta;
fprintf(' - epsilon_init : %g, dG : %g, eta : %g, maxiter : %g\n', epsilon_init, dGp, eta, maxiter);
fprintf('\n\n');
acc_interval = 5;

% parameter settings
h = [size(X,1);h(:);size(Y,1)];
L = numel(h);
G = cell(L-1);

for l = 1:L-1
    G{l} = rand(h(l),h(l+1)) * epsilon_init * 2 - epsilon_init;
end

oZ = cell(L); % for online
oV = cell(L); % for online
V = cell(L); % for calculating Error
V{1} = X;

m = size(X,2);
mse = zeros(1,maxiter);
acc = zeros(1,floor(maxiter/acc_interval));

% iteration
for iter = 1:maxiter
    fprintf('%d iterations..', iter);
    
%  for each data(online learning)
    for n = 1:m

        oV{1} = X(:,n); % #(1) x 1
%     forward
        for l = 2:L
            oZ{l} = G{l-1}'*oV{l-1}; % #(l) x 1
            oV{l} = active(G{l-1}'*oV{l-1}); % #(l) x 1
        end
    
%     backward
        
        GxD = oV{L}-Y(:,n); % #(L) x 1
        for l = L-1:-1:1
            df = active_diff(oZ{l+1}); % #(l+1) x 1
            D = df.*(GxD); % #(l+1) x 1
            
            oVxD = oV{l}*D'; % #(l) x #(l+1)
            dG = zeros(size(oVxD)); % #(l) x #(l+1)
            dG(oVxD > dGm) = dGm; % 조건을 D>0으로 두게되면 oV 크기에 따른 차등이 사라져 자유도가 급감한다. 따라서 oVxD > dGm으로 둔다.
           dG(oVxD < -dGp) = -dGp; % D<0또한 위와 같다. 따라서 OVxD < -dGp로 둔다.
%            dG = oVxD;
            
            G{l} = G{l} - (1/m)*eta*dG; % #(l) x #(l+1)
            
            GxD = G{l}*D; % #(l) x 1
        end
    end
    
    
% calculate mse
    for l = 2:L
        V{l} = active(G{l-1}'*V{l-1}); % #(l) x m
    end
    E = V{L}-Y;
    mse(iter) = mean(dot(E(:),E(:)));

    fprintf('mse : %d\n', mse(iter));
    
% set acc logs
    if mod(iter,acc_interval)==0
        model.W = G;
        pred_Y = mlpPred(model, X);
        [~, pred_y] = max(pred_Y, [], 1);
        [~, y] = max(Y, [], 1);
        
        acc(floor(iter/acc_interval)) = mean(double(pred_y == y)) * 100;
        fprintf('[iter : %d]Training Set Accuracy: %f\n', iter, acc(floor(iter/acc_interval)));
    end
    
end
mse = mse(1:iter);
model.W = G;

% draw graph
figure
ax1 = gca;
hold on
plot(1:maxiter,mse,'b');
ax2 = axes('Position',get(ax1,'Position'),...
       'YAxisLocation','right',...
       'Color','none',...
       'YLim', [0,100],...
       'XColor','k','YColor','k');
linkaxes([ax1 ax2],'x');
hold on
%plot(x,y3,'Parent',ax2);
plot(acc_interval:acc_interval:iter,acc,'r');

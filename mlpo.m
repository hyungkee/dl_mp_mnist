function [model, mse] = mlpo(X, Y, h, options)
% 이 함수는 online learning을 위한 함수이다.

fprintf('\n\n');
fprintf('[online learning]\n');

% option읽기
active = options.active;
active_diff = options.active_diff;
epsilon_init = options.epsilon_init;
minG = options.minG;
maxG = options.maxG;
stepG = options.stepG;
maxiter = options.MaxIter;
acc_interval = 5; % for monitoring mse

% parameter settings
h = [size(X,1);h(:);size(Y,1)];
L = numel(h);
Gp = cell(L-1);
Gm = cell(L-1);

for l = 1:L-1
    Gp{l} = rand(h(l),h(l+1)) * (maxG - minG) * epsilon_init + minG;
    Gm{l} = rand(h(l),h(l+1)) * (maxG - minG) * epsilon_init + minG;
end

oZ = cell(L); % for online
oV = cell(L); % for online
V = cell(L); % for calculating Error
V{1} = X;

m = size(X,2);
mse = zeros(1,maxiter);
acc = zeros(1,floor(maxiter/acc_interval)); % for monitering mse

% iteration
for iter = 1:maxiter
    fprintf('%d iterations..', iter);
    
%  for each data(online learning)
    for n = 1:m

        oV{1} = X(:,n); % #(1) x 1
%     forward
        for l = 2:L
            oZ{l} = (Gp{l-1}'-Gm{l-1}')*oV{l-1}; % #(l) x 1
            oV{l} = active((Gp{l-1}'-Gm{l-1}')*oV{l-1}); % #(l) x 1
        end
    
%     backward
        
        GxD = oV{L}-Y(:,n); % #(L) x 1
        for l = L-1:-1:1
            df = active_diff(oZ{l+1}); % #(l+1) x 1
            D = df.*(GxD); % #(l+1) x 1
            
            oVxD = oV{l}*D'; % #(l) x #(l+1)
            dG = zeros(size(oVxD)); % #(l) x #(l+1)
            dG(oVxD > 0) = stepG; % 조건을 D>0으로 두게되면 oV 크기에 따른 차등이 사라져 자유도가 급감한다. 따라서 oVxD > dGm으로 둔다.
            dG(oVxD < 0) = -stepG; % D<0또한 위와 같다. 따라서 OVxD < -dGp로 둔다.
%            dG = oVxD;

%TODO : Gp, Gm을 위한 조정 알고리즘을 만들어야 한다. 함수로 만들자.
            Gp{l} = Gp{l} - dG; % #(l) x #(l+1)
            
            GxD = (Gp{l}-Gm{l})*D; % #(l) x 1
        end
    end
    
    
% calculate mse
    for l = 2:L
        V{l} = active((Gp{l-1}'-Gm{l-1}')*V{l-1}); % #(l) x m
    end
    E = V{L}-Y;
    mse(iter) = mean(dot(E(:),E(:)));

    fprintf('mse : %d\n', mse(iter));
    
% set acc logs
    if mod(iter,acc_interval)==0

        model.W = cell(L-1);
        for l = 1:L-1
            model.W{l} = Gp{l} - Gm{l};
        end
        
        pred_Y = mlpPred(model, X, active);
        [~, pred_y] = max(pred_Y, [], 1);
        [~, y] = max(Y, [], 1);
        
        acc(floor(iter/acc_interval)) = mean(double(pred_y == y)) * 100;
        fprintf('[iter : %d]Training Set Accuracy: %f\n', iter, acc(floor(iter/acc_interval)));
    end
    
end
mse = mse(1:iter);
model.W = cell(L-1);
for l = 1:L-1
    model.W{l} = Gp{l} - Gm{l};
end

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

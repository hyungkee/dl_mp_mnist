function [model, mse] = mlp(X, Y, h, options)

fprintf('\n\n');
fprintf('[offline learning]\n');

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

Z = cell(L);
V = cell(L);
V{1} = X;

m = size(X,2);
mse = zeros(1,maxiter);
acc = zeros(1,floor(maxiter/acc_interval));

for iter = 1:maxiter
    fprintf('%d iterations..', iter);
    
%     forward
    for l = 2:L
        Z{l} = G{l-1}'*V{l-1};
        V{l} = active(Z{l});
    end
    
%     backward
    E = V{L}-Y;
    mse(iter) = mean(dot(E(:),E(:)));

    GxD = V{L}-Y; % #(L) x m
    for l = L-1:-1:1
        
        df = active_diff(Z{l+1}); % #(l+1) x m
        D = df.*(GxD); % #(l+1) x m
        
        qV = ones(size(V{l})); % #{l} x m. quantized V
%        qV = zeros(size(V{l})); % #{l} x m. quantized V
%        qV(V{l}>0) = 1; % V{l}>=0, 각 곱을 위해 0과 1로 이진화. 0.01은 threshold
        qV = V{l};
        
        qD = zeros(size(D)); % #{l+1} x m. quantized D
        qD(D >0) = dGm;
        qD(D <0) = -dGp;
        
        dG = qV*qD'; % {#(l) x m} * {m x #(l+1)} = #(l) x #(l+1)
        G{l} = G{l} - (1/m)*eta*dG;
        
        GxD = G{l}*D; % #(l) x m
    end
    
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
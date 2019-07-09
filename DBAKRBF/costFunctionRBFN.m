function [E, grad] = costFunctionRBFN(parm , y_one_hot , n , Centers , X_train , K , precomputedtw)
% lambda = 0.0001;
% parm : 需要训练的所有参数
% n : 神经元数目 the Number of neurons
% K : 类别数， 输出层神经元数目 classes
% D : 向量的维度 Dimension of the center vector 
% N : 样本数目 The sample size
% Centers : 神经元中心 ， 每行表示一个中心 n-by-D  the centers of neurons, each row represents a center
% phi : 激活值 , 每行表示一个样本的所有神经元的激活 N-by-n+1 Each row represents the activation of all neurons in a sample
% beta : 每个神经元的超参数 n-by-1 the Superparameters of neurons
% alpha : 输出层权重，每行表示输出层神经元的所有连接权重 K-by-n+1  weight of the output layer
% X_train : 训练样本 ，每行表示一个训练样本 N-by-D
% y_score : 每个样本预测为第k类的分数 ，N-by-K
% y_one_hot : 样本标签 one hot 表示 , N-by-K
% y : 样本标签 , N-by-1
% E : loss , 1-by-1
% gradAlpha : Alpha 的 梯度 ， K-by-n+1
% gradBeta : Beta 的 梯度 ， n-by-1

[N,D] = size(X_train);
alpha = parm(1:(n + 1) * K , :);
alpha = reshape(alpha , K , n + 1);
beta = parm((n + 1) * K + 1:end,:);

% 计算激活值

[phi,graddist] = RBF_calcAGDTWKernel(X_train , Centers ,beta, precomputedtw);
% nstd = std(phi,[],2);
% nmax = mean(phi,2);
% phi = phi - nmax;
% phi = phi ./ nstd;

% phi = zeros(N, n);
% for i = 1 : N
%     input = X_train(i, :);
%     z = getRBFActivations(Centers, beta, input);
%     phi(i, :) = z';
% end

if max(phi(:)) == inf
    fprintf('phi 存在 inf\n');
end
% if max(graddist(:)) == inf
%     fprintf('graddist 存在 inf\n');
% end

phi = [ones(N, 1), phi];

% 计算y_score
y_score = phi*alpha';
if max(y_score(:)) == inf
    fprintf('y_score 存在 inf\n');
end
% 计算loss
% E = sum(sum((y_score - y_one_hot).^2));
% E = E / (2 * N);

% 计算 softmax
% nmax = mean(y_score,[],1);
% 所有样本归一化
% nstd = std(y_score,[],1);
% nmax = mean(y_score);
% nmax = max(y_score,[],2);
% y_score = y_score - nmax;
% y_score = bsxfun(@minus,y_score,nmax);
% y_score = y_score ./ nstd;

% 样本单独归一化
% nstd = std(y_score,[],2);
% nmax = mean(y_score,2);
% y_score = bsxfun(@minus,y_score,nmax);
% y_score = y_score ./ nstd;
% y_score = y_score ./ n;
% y_score = bsxfun(@rdivide,y_score,nstd);
% y_score = y_score ./ mean(y_score,2);
y_score = y_score - max(y_score , [] , 2);
numerator = exp(y_score);
% numerator(numerator > 1e7) = 1e7;
denominator = sum(numerator,2); 
denominator(denominator == 0) = 1;
% y_softmax = bsxfun(@rdivide,numerator,denominator);
y_softmax = numerator ./ denominator;

if max(numerator(:)) == inf
    fprintf('numerator 存在 inf\n');
end
if max(denominator(:)) == inf
    fprintf('denominator 存在 inf\n');
end
if max(y_softmax(:)) == inf
    fprintf('y_softmax 存在 inf\n');
end

% 计算 cross entropy
t = y_one_hot;
y = y_softmax;
y = max(min(y,1-eps),eps);
t = max(min(t,1),0);
perfs = -t.*log(y);

lambda = 0;

E = sum(sum(perfs,2)) / N + lambda * sum(sum(alpha(:,2:end) .*alpha(:,2:end))) + lambda * sum(sum(beta .*beta));

if max(y_softmax(:)) == inf
    fprintf('perfs 存在 inf\n');
end
if max(E(:)) == inf
    fprintf('E 存在 inf\n');
end

% 计算 alpha 的梯度
normgradalpha = 2 * lambda .* alpha;
normgradalpha(:,1) = 0;
gradAlpha = (y_softmax - y_one_hot)'*phi / N + normgradalpha;
if max(gradAlpha(:)) == inf
    fprintf('gradAlpha 存在 inf\n');
end

normgradbeta = 2 * lambda .* beta;
% graddist = -phi(:,2:end) .* (pdist2(X_train , Centers,'euclidean').^2);
gradBeta = sum(((y_softmax - y_one_hot)*alpha(:,2:end)).*graddist,1)'/N + normgradbeta;
if max(gradBeta(:)) == inf
    fprintf('gradBeta 存在 inf\n');
end
% gradBeta = zeros(size(gradBeta));
% gradAlpha = zeros(size(gradAlpha));
grad = [reshape(gradAlpha , [] , 1) ; gradBeta];
end

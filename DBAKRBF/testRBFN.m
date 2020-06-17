function [accuracy ,numRight ,m] = testRBFN(Centers, betas, Theta, test_x, y , precomputedtw)
% 计算DBAK-RBF网络的输出。
% Centers : 神经元中心 ， 每行表示一个中心 n-by-D
% beta : 每个神经元的超参数 n-by-1
% Theta  :每个神经元的输出权重 n-by-1
% input:DBAK-RBF网络的输入
% accuracy:正确率 accuracy = numRight/m
% numRight:正确数量
% m:样例总数

% 计算每个神经元的激活值
%   phis = getRBFActivations(Centers, betas, input)';
    [m,~] = size(test_x);
    if nargin < 6
        phis = RBF_calcAGDTWKernel(test_x,Centers,betas);
    else 
     phis = RBF_calcAGDTWKernel(test_x,Centers,betas , precomputedtw);
    end
%     phis(find(phis > 1e7)) = 1e7;
    % Add a 1 to the beginning of the activations vector for the bias term.
    phis = [ones(m, 1) phis];
    
    % Multiply the activations by the weights and take the sum. Do this for
	% each category (output node). The result is a column vector with one row
	% per output node.
	%
	%   Theta = centroids x categories	  Theta' = categories x centroids
	%    phis = centroids x 1
	%       z = Theta' * phis = categories x 1
    z = phis*Theta';
    
    [~ , predictcategory] = max(z, [] ,2);
    numRight = sum(predictcategory == y);
    accuracy = numRight / m;
    % nmax = mean(y_score);
    % y_score = bsxfun(@minus,y_score,nmax);
%     numerator = exp(z);
%     denominator = sum(numerator,2); 
%     denominator(denominator == 0) = 1;
%     z = bsxfun(@rdivide,numerator,denominator);
        
end
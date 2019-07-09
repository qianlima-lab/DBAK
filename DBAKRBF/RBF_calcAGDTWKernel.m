function [test_kernel , graddist] = RBF_calcAGDTWKernel(data , mu  ,gamma , test_path_dtw)
test_x = data;
trainning_x = mu;

n_test = size(test_x , 1);
n_trainning = size(trainning_x , 1);

if nargin < 4
    [~ , test_path_dtw] = RBF_calcDtw(test_x , trainning_x );
end

dist_test = zeros(n_test , n_trainning);
graddist = zeros(n_test , n_trainning);
D = size(test_x , 2);
for i = 1 : n_test
    for j = 1 : n_trainning
        x_new = test_path_dtw{i ,j}{1};
        y_new = test_path_dtw{i ,j}{2};
        n_path = size(x_new , 2);
%         for k = 1 : n_path
%             dist_test( i ,j) = dist_test( i, j ) + exp( -gamma(j)*(x_new(k) - y_new(k))^2 );
%             graddist(i,j) = graddist(i,j) + exp( -gamma(j)*norm(x_new(k) - y_new(k))^2 ) * (-(x_new(k) - y_new(k))^2);
%         end
        a = -(x_new - y_new).^2;        
        h = exp(gamma(j).*(a));
        dist_test( i ,j) = sum(h);
        graddist(i,j) = sum(h .* (a));        
        dist_test( i ,j) =  dist_test( i ,j) /n_path;
        graddist(i,j) = graddist(i,j) / n_path;
%         dist_test( i ,j) =  dist_test( i ,j) /n_path * (D / n_path);
%         graddist(i,j) = graddist(i,j) / n_path* (D / n_path);
%         dist_test( i ,j) =  dist_test( i ,j) /D;
%         graddist(i,j) = graddist(i,j) / D;
    end
end

test_kernel =  dist_test;
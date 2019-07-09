clear;tic;
% Add the subdirectories to the path.
addpath('DBAKRBF');
addpath('fmin_adam');
addpath('kMeans');

rng(1);
isScale = 6;
batch_size = 2;
dataname =  'DistalPhalanxTW';
learningrate = 0.01;
maxiter = 5E4;
%icenter = 27;

fprintf('*******************数据集:%s*******************\n',dataname);
fprintf('----------放缩方式:%d----------\n',isScale);
[X ,y , test_x , test_y] = readData(dataname,isScale); 

numCats = size(unique(y), 1);

nglobal =38;
nlocal =5;
icenter = nlocal * numCats + nglobal;

%%calculate the center 
%icenter = 120;
%maxnLocal = size(y,1);
%for c = 1:numCats
 %   maxnLocal = min(maxnLocal,sum(y == c));
%end
%nlocal = floor(icenter / numCats);
%if nlocal >= maxnLocal
%    nlocal = maxnLocal;    
%end
%nglobal = icenter - nlocal * numCats;

fprintf('......中心数:%d.......\n',icenter);
disp('Training the DBAK-RBF...');            
[iter ,trainacc, testacc,Theta,betas,Centers] = exp_trainRBFN(X, y,test_x , test_y,nglobal,nlocal,learningrate,maxiter,batch_size,1);
toc;
size(X,1)
%[iter ,trainacc, testacc,Theta,betas,Centers] = exp_trainRBFN(X, y,test_x , test_y,nglobal,nlocal,0.05,10000,32,Theta,betas,Centers);
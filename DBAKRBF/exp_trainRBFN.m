function [iter , trainacc, testacc,Theta,betas,Centers] = exp_trainRBFN(X_train, y_train,test_x , test_y,nglobal,nlocal,learningrate,maxiter,batch_size,iTheta,ibetas,iCenters)

    numCats = size(unique(y_train), 1);% 类别数
    m = size(X_train, 1);% 样本数
    
    if(nargin < 12)
        % 确定标签为1~numCats
        if (any(y_train == 0) || any(y_train > numCats))
            error('Category values must be non-zero and continuous.');
        end
        
        fprintf('Number of Local center per class:%d\n',nlocal);
        
        Centers = []; 
        % local center
        if nlocal > 0
            for (c = 1 : numCats)
                % Select the training vectors for category 'c'.
                Xc = X_train((y_train == c), :);
                init_Centroids = Xc(1:min(nlocal,sum(y_train == c)), :);
                nglobal = nglobal + nlocal - min(nlocal,sum(y_train == c));
                [memberships_c,Centroids_c] = kMeans(Xc,init_Centroids,100);
                % do dba for every cluster
                for i = 1 : size(Centroids_c, 1)
                    Xc_cluster = Xc((memberships_c == i), :);
                    if(size(Xc_cluster , 1) > 0)
                        Centroids_c(i,:) = DBA(Xc_cluster);
                    end       
                end        
                Centers = [Centers; Centroids_c];
            end
        end
        
        fprintf('Number of Global center:%d\n',nglobal);
        % global center
        if nglobal > 0
            Xc = X_train
            %[memberships_c,Centroids_c] = kmeans(Xc,nglobal);
            init_Centroids = Xc(1:nglobal, :);
            [memberships_c,Centroids_c] = kMeans(Xc,init_Centroids,100);
            % do dba for every cluster
            for i = 1 : size(Centroids_c, 1)
                Xc_cluster = Xc((memberships_c == i), :);
                if(size(Xc_cluster , 1) > 0)
                    Centroids_c(i,:) = DBA(Xc_cluster);
                end       
            end        
            Centers = [Centers; Centroids_c];
        end  
            
    else
        % 已存在
        Centers = iCenters;
        Theta = iTheta(:);
        betas = ibetas(:);
        numRBFNeurons = size(Centers, 1); 
    end
    M= [];
    y_one_hot = zeros(m , numCats);
    for i = 1 : numCats
        y_one_hot(:,i) = (y_train == i);
    end
        
    sOpt = optimset('fmin_adam');
    sOpt.MaxIter = maxiter;
    learning_rate = learningrate;
    
      numRBFNeurons = size(Centers, 1);   
    Theta = rand((numRBFNeurons + 1) * numCats, 1);
    betas = rand(numRBFNeurons , 1);
          
    
    fprintf('Train DBAK-RBF\n'); 
   % 计算真实的样本标签 ， 这个可以放在函数外面进行
    phi = [Theta ; betas];       
    [best, fval, exitflag, output] = fmin_adam(@(phi , X_input , y_input_onehot , x_precomputedtw)costFunctionRBFN(phi,  y_input_onehot , numRBFNeurons , Centers , X_input , numCats , x_precomputedtw),phi, learning_rate,[],[],[],[],sOpt,numRBFNeurons,numCats , X_train, y_train,test_x , test_y , Centers,batch_size);
    fprintf('寻优最优解:%.1f%% , 迭代次数:%d ',best.testacc*100 , best.iteration);
    phi = best.x;
    Theta = phi(1:(numRBFNeurons + 1) * numCats , :);
    Theta = reshape(Theta , numCats , (numRBFNeurons + 1));
    betas = phi((numRBFNeurons + 1) * numCats + 1:end,:);
    testacc = best.testacc;
    trainacc = best.trainacc;
    iter = best.iteration;
    
end

function [ trainning_x ,trainning_y , test_x , test_y ,trainning_data , test_data] = readData(filename, isScale)
trainning_data = load(  strcat('datasets/',filename,'/',filename,'_TRAIN'));
trainning_x = trainning_data(: , 2:end);
trainning_y = trainning_data(: , 1);

test_data = load(  strcat('datasets/',filename,'/',filename,'_TEST'));
test_x = test_data(: , 2:end);
test_y = test_data(: , 1);

switch isScale
    case 0
    case 4
        for i = [1:size(trainning_x,1)]
            trainning_x(i,:) = mapminmax(trainning_x(i,:),0,1);
        end
    case 5
        for i = [1:size(trainning_x,1)]
            trainning_x(i,:) = mapminmax(trainning_x(i,:),-1,1);
        end
    case 6
        for i = [1:size(trainning_x,1)]
            trainning_x(i,:) = mapstd(trainning_x(i,:),0,1);
        end
    case 3
        [trainning_x psx] = mapminmax(trainning_x' , 0, 1);
        trainning_x = trainning_x';
        test_x = mapminmax('apply',test_x',psx);
        test_x = test_x';
    case 2
        [trainning_x psx] = mapminmax(trainning_x' , -1, 1);
        trainning_x = trainning_x';
        test_x = mapminmax('apply',test_x',psx);
        test_x = test_x';
    case 1
        [trainning_x psx] = mapstd(trainning_x' , 0 ,1);
        trainning_x = trainning_x';
        test_x = mapstd('apply',test_x',psx);
        test_x = test_x';
end

y = [trainning_y ; test_y];
[~, loc] = ismember(y, unique(y));
trainning_y = loc(1:size(trainning_y , 1) , 1);
test_y = loc(size(trainning_y , 1)+1:end , 1);

end
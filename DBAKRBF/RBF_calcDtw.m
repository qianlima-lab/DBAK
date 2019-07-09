function [dist_dtw , path_dtw] = RBF_calcDtw(data_x , data_y )

    total_x = size(data_x , 1);
    total_y = size(data_y , 1);

    dist_dtw = zeros(total_x , total_y);
    path_dtw = cell(total_x , total_y);
    for i = 1 : total_x,    
        x = data_x(i,:);
        for j = 1: total_y,
            y = data_y(j , :);
            [dist_dtw(i,j) , ix , iy] = dtw(x ,y,'squared');    
            path_dtw{i , j} = {x(:,ix); y(:,iy)};
        end    
    end


    